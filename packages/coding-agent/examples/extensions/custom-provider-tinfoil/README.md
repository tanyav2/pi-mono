# Tinfoil Provider Extension

Registers Tinfoil as a custom pi model provider. The extension uses the Tinfoil JavaScript SDK directly so requests are verified and encrypted before being sent to Tinfoil enclaves.

## Setup

```bash
# From the pi-mono repository root:
npm install

export TINFOIL_API_KEY="<your-tinfoil-api-key>"
./pi-test.sh -e ./packages/coding-agent/examples/extensions/custom-provider-tinfoil/index.ts --model tinfoil/gpt-oss-120b
```

If you are using an installed or built `pi` binary, use the same extension and model arguments with `pi` instead of `./pi-test.sh`.

You can also load the extension and choose a model from `/model`:

```bash
./pi-test.sh -e ./packages/coding-agent/examples/extensions/custom-provider-tinfoil/index.ts
```

## Models

The extension registers Tinfoil Chat Completions models under `tinfoil/...`:

- `tinfoil/deepseek-v4-pro`
- `tinfoil/glm-5-1`
- `tinfoil/kimi-k2-6`
- `tinfoil/gemma4-31b`
- `tinfoil/qwen3-vl-30b`
- `tinfoil/gpt-oss-120b`
- `tinfoil/llama3-3-70b`

It also registers Tinfoil Responses API models under `tinfoil-responses/...`:

- `tinfoil-responses/deepseek-v4-pro`
- `tinfoil-responses/qwen3-vl-30b`
- `tinfoil-responses/gpt-oss-120b`

For example:

```bash
./pi-test.sh -e ./packages/coding-agent/examples/extensions/custom-provider-tinfoil/index.ts --model tinfoil-responses/gpt-oss-120b
```
