# craft-ai-fantastico

A [FANTASTICO](https://fantastico.cc) account as a provider for the
[Craft AI layer](https://github.com/storytold/craft-ai-integrations). FANTASTICO holds one prepaid
credit with spending caps and fronts every major LLM (Claude, GPT, Gemini, Llama…). With this
adapter registered, a person adds their account key once, in the launcher or in any Craft app, and
every app that registers the adapter can run its text tools on any of those models.

## Why an adapter and not a custom connection

A FANTASTICO account key is never sent to a model. `fantastico-sdk` trades it for a short-lived
runtime key capped to part of the credit, renews that key before it expires, fits `max_tokens` to
the cap, and sends the chat request straight to the provider behind the account, so the account
service never sees a prompt. A custom OpenAI-compatible connection cannot do the exchange, and the
layer's declarative adapter rules out token refresh and streaming by design. So this crate
implements `craft_ai_core::Provider` on top of the SDK.

## Using it

```rust,ignore
let mut factory = ProviderFactory::standard();
factory.register(craft_ai_fantastico::PROVIDER_NAME, craft_ai_fantastico::build);
let client = AiClient::new(app, shared_store, keychain).with_factory(factory);
```

A connection saved with provider `fantastico` and an account key (`ck-fant-…`) then works in that
app. Its `endpoint`, if set, replaces the account service address (`https://fantastico.cc/api`).
An app that does not register the adapter reports the connection as one it cannot run yet.

| Capability | Notes |
|---|---|
| `text.complete` | streaming, tool calls, structured output, cost per request (US dollars) |

Not yet: images, speech, embeddings, image input to chat, cost estimates before a job. Each is
refused with a clear `Unsupported` error before anything is sent.

`build_titled("WordCraft")` registers the same adapter with the app's name in the provider's usage
reports.

### Conversations (this adapter's extension)

The Craft AI contract has no input for an earlier assistant reply or a tool result, which an agent
loop needs, and its `ToolCall` event has no call id. Until it has both, this adapter takes a `turn`
input (`craft_ai_fantastico::TURN`): one chat message as JSON, any of system, user, assistant (with
`tool_calls`) or tool (with `tool_call_id`). A request with turns also gets the whole reply as one
`application/json` output labelled `completion` (`craft_ai_fantastico::COMPLETION`), ids and finish
reason included. Other adapters refuse turns as `Unsupported`. WordCraft's GeenioAI runs its agent
loop this way.

## How it behaves

- **Models** come from the account service and double as the connection test: the key is checked
  and nothing is minted or spent. Capabilities are read from each entry (`supported_parameters`,
  `output_modalities`), not guessed from ids; image-only models are left out.
- **Jobs** run on worker threads. Streamed text goes into the job as it arrives; a cancelled
  stream is closed at its next line, which ends the provider's work.
- **Keys**: each concurrent job slot keeps one SDK client and so one runtime key, reused from job to
  job. `max_active_jobs` (default 2) bounds how much credit the open keys reserve; one job more is
  refused as `RateLimited` rather than queued.
- **Errors** map FANTASTICO's codes to the layer's kinds: an empty account is `Quota`, a refused key
  `Auth`, an unreachable service `Network`, an over-long or refused prompt `InvalidRequest`.
- **Safety**: the account key goes only to the account service and the runtime key only to the
  provider; redirects are never followed; TLS and proxy rules come from `craft-ai-http`; replies,
  stream lines, event backlogs and model lists are capped; no `unwrap`, `expect`, `panic` or
  `unsafe` in shipped code.

## Status

The Craft AI contract (`craft-ai-core`) is **provisional** until its review in
`storytold/craft-libs` finishes, so this crate pins one revision of `craft-ai-integrations` and
will follow it. Nothing here is a final API.

What has been verified: every behaviour above against a scripted account service and a scripted
provider (`tests/conformance.rs`, including the layer's own conformance checks). Not verified yet:
a real FANTASTICO account, and an app running a tool through the shared binding.

## Building and testing

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all --check
```

## Licence

MIT or Apache-2.0, at your option (`LICENSE-MIT`, `LICENSE-APACHE`).
