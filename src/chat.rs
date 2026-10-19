//! Chat completions: the request body built from a [`Request`], and the reply turned into job
//! events. Pure (no I/O), so it is tested without a server.
//!
//! Inputs become messages in the order given, except that `system` inputs come first: `system`
//! becomes a system message, `prompt` and `context` become user messages. Only text inputs are
//! sent; anything else is refused rather than dropped.
//!
//! Params passed through (anything else is ignored, so a param can never steer the URL, a key or
//! the account): `temperature`, `top_p`, `max_tokens`, `seed`, `stop`, `frequency_penalty`,
//! `presence_penalty`, `tools`, `tool_choice` and `response_format`. `user` is deliberately not
//! among them: FANTASTICO refuses to attribute spending to another account. Giving
//! `response_format` makes the request *structured*: the reply must be JSON and arrives as one
//! `application/json` output instead of text deltas.

use craft_ai_core::job::{Event, Output};
use craft_ai_core::{AiError, ErrorKind, InputRole, Payload, Request, Usage};
use serde_json::{Map, Value, json};

use crate::stream::MAX_TOOL_CALLS;

const PASS_THROUGH: &[&str] = &[
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "tools",
    "tool_choice",
    "response_format",
];

/// A request ready to send.
#[derive(Debug)]
pub(crate) struct ChatPlan {
    pub body: Value,
    pub stream: bool,
    pub structured: bool,
}

pub(crate) fn plan(model: &str, req: &Request) -> Result<ChatPlan, AiError> {
    let mut system = Vec::new();
    let mut messages = Vec::new();
    for input in &req.inputs {
        let is_system = input.role == InputRole::System;
        if !is_system && input.role != InputRole::Prompt && input.role != InputRole::Context {
            return Err(AiError::unsupported(format!(
                "chat models do not take a {} input",
                input.role
            )));
        }
        let Payload::Text { text } = &input.payload else {
            return Err(AiError::unsupported(
                "this adapter sends text inputs to chat models only",
            ));
        };
        if is_system {
            system.push(json!({"role": "system", "content": text}));
        } else {
            messages.push(json!({"role": "user", "content": text}));
        }
    }
    // System messages first whatever order they came in: some models refuse one after a user turn.
    system.append(&mut messages);
    if system.is_empty() {
        return Err(AiError::invalid("a chat request needs a prompt"));
    }
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("messages".into(), Value::Array(system));
    for key in PASS_THROUGH {
        if let Some(v) = req.params.get(*key) {
            body.insert((*key).to_owned(), v.clone());
        }
    }
    if req.stream {
        body.insert("stream".into(), json!(true));
        body.insert("stream_options".into(), json!({"include_usage": true}));
    }
    Ok(ChatPlan {
        body: Value::Object(body),
        stream: req.stream,
        structured: req.params.contains_key("response_format"),
    })
}

/// What a finished reply held.
#[derive(Debug, Default)]
pub(crate) struct Completion {
    pub text: String,
    pub tool_calls: Vec<(String, String)>,
    pub usage: Option<Usage>,
}

/// Token counts and what the request cost. The provider behind FANTASTICO is always OpenRouter,
/// whose `usage.cost` is the charge in US dollars, so it is kept.
fn usage_from(v: &Value) -> Option<Usage> {
    let u = v.get("usage").filter(|u| u.is_object())?;
    let n = |keys: &[&str]| keys.iter().find_map(|k| u.get(*k).and_then(Value::as_u64));
    let cost = u
        .get("cost")
        .and_then(Value::as_f64)
        .filter(|c| c.is_finite() && (0.0..1.0e9).contains(c))
        .map(dollars_to_micros);
    Some(Usage {
        input_units: n(&["prompt_tokens", "input_tokens"]),
        output_units: n(&["completion_tokens", "output_tokens"]),
        cost_micros: cost,
        currency: cost.map(|_| "USD".to_owned()),
    })
}

/// Dollars to millionths, rounded. The caller has checked the range.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn dollars_to_micros(dollars: f64) -> u64 {
    (dollars * 1_000_000.0).round() as u64
}

/// Reads a `chat.completion` object, whole or assembled from a stream.
pub(crate) fn parse_completion(v: &Value) -> Result<Completion, AiError> {
    let message = v
        .pointer("/choices/0/message")
        .ok_or_else(|| AiError::new(ErrorKind::Provider, "the reply had no choices"))?;
    let text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(MAX_TOOL_CALLS)
        .map(|call| {
            let name = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let args = call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            (name.to_owned(), args.to_owned())
        })
        .collect();
    Ok(Completion {
        text,
        tool_calls,
        usage: usage_from(v),
    })
}

/// The events a finished reply adds. A streamed plain reply already delivered its text as deltas,
/// so only its tool calls are left; a structured reply becomes one JSON output.
pub(crate) fn events_for(
    c: &Completion,
    structured: bool,
    streamed: bool,
) -> Result<Vec<Event>, AiError> {
    let mut events = Vec::new();
    if structured {
        let parsed: Value = serde_json::from_str(c.text.trim()).map_err(|_| {
            AiError::new(
                ErrorKind::Provider,
                "the model was asked for structured output but did not return valid JSON",
            )
        })?;
        let bytes = serde_json::to_vec(&parsed)
            .map_err(|_| AiError::new(ErrorKind::Internal, "could not encode the JSON reply"))?;
        events.push(Event::Output(Output {
            payload: Payload::inline("application/json", bytes)?,
            label: None,
        }));
    } else if !streamed && !c.text.is_empty() {
        events.push(Event::TextDelta {
            text: c.text.clone(),
        });
    }
    for (name, arguments) in &c.tool_calls {
        events.push(Event::ToolCall {
            name: name.clone(),
            arguments: arguments.clone(),
        });
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use craft_ai_core::CapabilityId;

    fn req() -> Request {
        Request::new(CapabilityId::parse("text.complete").unwrap())
    }

    #[test]
    fn inputs_become_messages_with_system_first() {
        let r = req()
            .with_input(InputRole::Prompt, Payload::text("hi"))
            .with_input(InputRole::System, Payload::text("be brief"))
            .with_input(InputRole::Context, Payload::text("more"));
        let p = plan("anthropic/claude-sonnet-5", &r).unwrap();
        let roles: Vec<_> = p.body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(roles, ["system", "user", "user"]);
        assert_eq!(p.body["model"], "anthropic/claude-sonnet-5");
        assert!(p.body.get("stream").is_none());
        assert!(!p.structured);
    }

    #[test]
    fn only_listed_params_pass_through_and_user_is_not_one() {
        let r = req()
            .with_input(InputRole::Prompt, Payload::text("hi"))
            .with_param("temperature", json!(0.2))
            .with_param("tools", json!([{"type": "function"}]))
            .with_param("response_format", json!({"type": "json_object"}))
            .with_param("user", json!("someone-else"))
            .with_param("model", json!("other/model"))
            .with_param("base_url", json!("http://elsewhere"));
        let p = plan("m", &r).unwrap();
        assert_eq!(p.body["temperature"], 0.2);
        assert!(p.body["tools"].is_array());
        assert!(p.structured);
        assert_eq!(p.body["model"], "m");
        assert!(p.body.get("user").is_none());
        assert!(p.body.get("base_url").is_none());
    }

    #[test]
    fn streaming_asks_for_usage() {
        let p = plan(
            "m",
            &req()
                .with_input(InputRole::Prompt, Payload::text("x"))
                .streaming(),
        )
        .unwrap();
        assert_eq!(p.body["stream"], true);
        assert_eq!(p.body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn non_text_and_odd_roles_are_refused_not_dropped() {
        let img = Payload::inline("image/png", vec![1]).unwrap();
        let e = plan("m", &req().with_input(InputRole::Prompt, img)).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Unsupported);
        let e = plan("m", &req().with_input(InputRole::Mask, Payload::text("x"))).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Unsupported);
        assert_eq!(
            plan("m", &req()).unwrap_err().kind,
            ErrorKind::InvalidRequest
        );
    }

    #[test]
    fn a_reply_with_tools_and_cost_is_read() {
        let v = json!({"choices":[{"message":{"content":"hello","tool_calls":[
            {"function":{"name":"get_time","arguments":"{\"tz\":\"UTC\"}"}}]}}],
            "usage":{"prompt_tokens":3,"completion_tokens":5,"cost":0.00014}});
        let c = parse_completion(&v).unwrap();
        assert_eq!(c.text, "hello");
        assert_eq!(
            c.tool_calls,
            [("get_time".into(), "{\"tz\":\"UTC\"}".into())]
        );
        let u = c.usage.unwrap();
        assert_eq!(
            (u.output_units, u.cost_micros, u.currency.as_deref()),
            (Some(5), Some(140), Some("USD"))
        );
        assert!(parse_completion(&json!({})).is_err());
        for bad in [json!(-1.0), json!("0.5"), json!(1.0e12)] {
            let v = json!({"choices":[{"message":{}}],"usage":{"cost":bad}});
            assert_eq!(
                parse_completion(&v).unwrap().usage.unwrap().cost_micros,
                None
            );
        }
    }

    #[test]
    fn events_depend_on_streaming_and_structure() {
        let c = Completion {
            text: "plain".into(),
            tool_calls: vec![("f".into(), "{}".into())],
            usage: None,
        };
        let whole = events_for(&c, false, false).unwrap();
        assert!(matches!(&whole[0], Event::TextDelta { text } if text == "plain"));
        let streamed = events_for(&c, false, true).unwrap();
        assert_eq!(streamed.len(), 1);
        assert!(matches!(&streamed[0], Event::ToolCall { .. }));
        assert_eq!(
            events_for(&c, true, false).unwrap_err().kind,
            ErrorKind::Provider
        );
        let json_reply = Completion {
            text: " {\"a\": 1} ".into(),
            ..Completion::default()
        };
        assert!(matches!(
            &events_for(&json_reply, true, true).unwrap()[0],
            Event::Output(_)
        ));
    }
}
