/**
 * Tinfoil Provider Extension
 *
 * Registers Tinfoil as a custom provider using the Tinfoil JavaScript SDK.
 *
 * Usage:
 *   export TINFOIL_API_KEY="<your-tinfoil-api-key>"
 *   ./pi-test.sh -e ./packages/coding-agent/examples/extensions/custom-provider-tinfoil/index.ts --model tinfoil/gpt-oss-120b
 */

import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Context,
	createAssistantMessageEventStream,
	type ImageContent,
	type Model,
	parseStreamingJson,
	type SimpleStreamOptions,
	type TextContent,
	type ThinkingContent,
	type ToolCall,
} from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { TinfoilAI } from "tinfoil";

const TINFOIL_PROVIDER = "tinfoil";
const TINFOIL_API = "tinfoil-chat-completions" as Api;
const TINFOIL_RESPONSES_PROVIDER = "tinfoil-responses";
const TINFOIL_RESPONSES_API = "tinfoil-responses" as Api;
const TINFOIL_BASE_URL = "https://inference.tinfoil.sh/v1/";
const MAX_TOKENS = 16384;

const MODEL_CONFIGS = [
	["deepseek-v4-pro", "DeepSeek V4 Pro", 800000, false, true, 1.5, 5.25],
	["glm-5-1", "GLM-5.1", 200000, false, true, 1.5, 5.25],
	["kimi-k2-6", "Kimi K2.6", 256000, true, true, 1.5, 5.25],
	["gemma4-31b", "Gemma 4 31B", 256000, true, true, 0.45, 1],
	["qwen3-vl-30b", "Qwen3-VL 30B", 256000, true, true, 1.25, 4],
	["gpt-oss-120b", "GPT-OSS 120B", 131000, false, true, 0.75, 1.25],
	["llama3-3-70b", "Llama 3.3 70B", 128000, false, false, 1.75, 2.75],
] as const;

const RESPONSES_MODEL_IDS = new Set<(typeof MODEL_CONFIGS)[number][0]>([
	"deepseek-v4-pro",
	"qwen3-vl-30b",
	"gpt-oss-120b",
]);

const RESPONSES_MODEL_CONFIGS = MODEL_CONFIGS.filter(([id]) => RESPONSES_MODEL_IDS.has(id));

type ChatMessageParam = TinfoilAI.Chat.Completions.ChatCompletionMessageParam;
type ChatContentPart = TinfoilAI.Chat.Completions.ChatCompletionContentPart;
type ChatTool = TinfoilAI.Chat.Completions.ChatCompletionTool;
type ChatUserContent = TinfoilAI.Chat.Completions.ChatCompletionUserMessageParam["content"];
type ResponsesInput = TinfoilAI.Responses.ResponseInput;
type ResponsesInputContent = TinfoilAI.Responses.ResponseInputContent;
type ResponsesTool = TinfoilAI.Responses.Tool;
type ResponsesCreateParamsStreaming = TinfoilAI.Responses.ResponseCreateParamsStreaming;
type ResponsesStreamEvent = TinfoilAI.Responses.ResponseStreamEvent;
type ResponsesOutputMessage = TinfoilAI.Responses.ResponseOutputMessage;
type ResponsesReasoningItem = TinfoilAI.Responses.ResponseReasoningItem;
type ResponsesFunctionToolCall = TinfoilAI.Responses.ResponseFunctionToolCall;
type ResponsesFunctionCallOutputItemList = TinfoilAI.Responses.ResponseFunctionCallOutputItemList;
type ResponsesUsage = TinfoilAI.Responses.ResponseUsage;
type ResponsesStatus = TinfoilAI.Responses.ResponseStatus;
type ResponseTextPhase = NonNullable<ResponsesOutputMessage["phase"]>;

interface TinfoilClientCache {
	key: string;
	client: TinfoilAI;
}

interface TokenUsage {
	prompt_tokens?: number;
	completion_tokens?: number;
	total_tokens?: number;
	prompt_cache_hit_tokens?: number;
	prompt_tokens_details?: {
		cached_tokens?: number;
		cache_write_tokens?: number;
	};
}

interface TinfoilChunk {
	id?: string;
	model?: string;
	usage?: TokenUsage;
	choices?: TinfoilChoice[];
}

interface TinfoilChoice {
	finish_reason?: string | null;
	usage?: TokenUsage;
	delta?: TinfoilDelta;
}

interface TinfoilDelta {
	content?: string | null;
	reasoning?: string | null;
	reasoning_content?: string | null;
	reasoning_text?: string | null;
	tool_calls?: TinfoilToolCallDelta[];
}

interface TinfoilToolCallDelta {
	index?: number;
	id?: string;
	function?: {
		name?: string;
		arguments?: string;
	};
}

interface StreamingToolCallBlock extends ToolCall {
	partialArgs?: string;
	streamIndex?: number;
}

interface StreamingResponsesToolCallBlock extends ToolCall {
	partialJson?: string;
}

interface ResponsesTextSignature {
	v: 1;
	id: string;
	phase?: ResponseTextPhase;
}

let tinfoilClientCache: TinfoilClientCache | undefined;

function getTinfoilClient(apiKey?: string): TinfoilAI {
	const key = apiKey ?? "";

	if (!tinfoilClientCache || tinfoilClientCache.key !== key) {
		tinfoilClientCache = {
			key,
			client: new TinfoilAI({ apiKey }),
		};
	}

	return tinfoilClientCache.client;
}

function createOutput(model: Model<Api>): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function updateUsage(output: AssistantMessage, usage: TokenUsage | undefined, model: Model<Api>): void {
	if (!usage) return;

	const input = usage.prompt_tokens ?? 0;
	const outputTokens = usage.completion_tokens ?? 0;
	const cacheRead = usage.prompt_tokens_details?.cached_tokens ?? usage.prompt_cache_hit_tokens ?? 0;
	const cacheWrite = usage.prompt_tokens_details?.cache_write_tokens ?? 0;
	const inputCost = (input / 1_000_000) * model.cost.input;
	const outputCost = (outputTokens / 1_000_000) * model.cost.output;
	const cacheReadCost = (cacheRead / 1_000_000) * model.cost.cacheRead;
	const cacheWriteCost = (cacheWrite / 1_000_000) * model.cost.cacheWrite;

	output.usage = {
		input,
		output: outputTokens,
		cacheRead,
		cacheWrite,
		totalTokens: usage.total_tokens ?? input + outputTokens,
		cost: {
			input: inputCost,
			output: outputCost,
			cacheRead: cacheReadCost,
			cacheWrite: cacheWriteCost,
			total: inputCost + outputCost + cacheReadCost + cacheWriteCost,
		},
	};
}

function updateResponsesUsage(output: AssistantMessage, usage: ResponsesUsage | undefined, model: Model<Api>): void {
	if (!usage) return;

	const cacheRead = usage.input_tokens_details?.cached_tokens ?? 0;
	const input = Math.max((usage.input_tokens ?? 0) - cacheRead, 0);
	const outputTokens = usage.output_tokens ?? 0;
	const inputCost = (input / 1_000_000) * model.cost.input;
	const outputCost = (outputTokens / 1_000_000) * model.cost.output;
	const cacheReadCost = (cacheRead / 1_000_000) * model.cost.cacheRead;

	output.usage = {
		input,
		output: outputTokens,
		cacheRead,
		cacheWrite: 0,
		totalTokens: usage.total_tokens ?? input + cacheRead + outputTokens,
		cost: {
			input: inputCost,
			output: outputCost,
			cacheRead: cacheReadCost,
			cacheWrite: 0,
			total: inputCost + outputCost + cacheReadCost,
		},
	};
}

function mapStopReason(finishReason: string): AssistantMessage["stopReason"] {
	if (finishReason === "tool_calls" || finishReason === "function_call") return "toolUse";
	if (finishReason === "length") return "length";
	if (finishReason === "stop") return "stop";
	return "error";
}

function mapResponsesStopReason(status: ResponsesStatus | undefined): AssistantMessage["stopReason"] {
	if (!status) return "stop";
	if (status === "completed" || status === "in_progress" || status === "queued") return "stop";
	if (status === "incomplete") return "length";
	return "error";
}

function getReasoningDelta(delta: TinfoilDelta): { field: string; text: string } | undefined {
	for (const field of ["reasoning_content", "reasoning", "reasoning_text"] as const) {
		const text = delta[field];
		if (text) return { field, text };
	}
	return undefined;
}

function removeStreamingScratch(output: AssistantMessage): void {
	for (const block of output.content) {
		delete (block as { partialArgs?: string }).partialArgs;
		delete (block as { streamIndex?: number }).streamIndex;
	}
}

function removeResponsesScratch(output: AssistantMessage): void {
	for (const block of output.content) {
		delete (block as { partialJson?: string }).partialJson;
	}
}

function encodeResponsesToolCallId(callId: string, itemId?: string | null): string {
	return itemId ? `${callId}|${itemId}` : callId;
}

function splitResponsesToolCallId(id: string): { callId: string; itemId?: string } {
	const separator = id.indexOf("|");
	if (separator === -1) return { callId: id };
	return {
		callId: id.slice(0, separator),
		itemId: id.slice(separator + 1) || undefined,
	};
}

function encodeTextSignature(id: string, phase?: ResponseTextPhase | null): string {
	const payload: ResponsesTextSignature = { v: 1, id };
	if (phase) payload.phase = phase;
	return JSON.stringify(payload);
}

function parseTextSignature(signature: string | undefined): { id: string; phase?: ResponseTextPhase } | undefined {
	if (!signature) return undefined;
	if (!signature.startsWith("{")) return { id: signature };

	try {
		const parsed = JSON.parse(signature) as Partial<ResponsesTextSignature>;
		if (parsed.v === 1 && typeof parsed.id === "string") {
			const phase = parsed.phase === "commentary" || parsed.phase === "final_answer" ? parsed.phase : undefined;
			return { id: parsed.id, phase };
		}
	} catch {
		return undefined;
	}

	return undefined;
}

function parseReasoningSignature(signature: string | undefined): ResponsesReasoningItem | undefined {
	if (!signature) return undefined;

	try {
		const parsed = JSON.parse(signature) as Partial<ResponsesReasoningItem>;
		if (parsed.type === "reasoning" && typeof parsed.id === "string") {
			return parsed as ResponsesReasoningItem;
		}
	} catch {
		return undefined;
	}

	return undefined;
}

function convertUserContent(content: string | (TextContent | ImageContent)[]): ChatUserContent {
	if (typeof content === "string") return content;
	const parts: ChatContentPart[] = [];

	for (const part of content) {
		if (part.type === "text") {
			parts.push({ type: "text", text: part.text });
		} else if (part.type === "image") {
			parts.push({
				type: "image_url",
				image_url: {
					url: `data:${part.mimeType};base64,${part.data}`,
				},
			});
		}
	}

	return parts;
}

function convertMessages(context: Context): ChatMessageParam[] {
	const messages: ChatMessageParam[] = [];

	if (context.systemPrompt) {
		messages.push({ role: "system", content: context.systemPrompt });
	}

	for (const message of context.messages) {
		if (message.role === "user") {
			messages.push({ role: "user", content: convertUserContent(message.content) });
		} else if (message.role === "assistant") {
			const content = message.content
				.filter((block) => block.type === "text")
				.map((block) => block.text)
				.join("");
			const toolCalls = message.content.filter((block) => block.type === "toolCall");

			if (!content && toolCalls.length === 0) continue;

			messages.push({
				role: "assistant",
				content: content || null,
				...(toolCalls.length > 0
					? {
							tool_calls: toolCalls.map((toolCall) => ({
								id: toolCall.id,
								type: "function" as const,
								function: {
									name: toolCall.name,
									arguments: JSON.stringify(toolCall.arguments),
								},
							})),
						}
					: {}),
			});
		} else if (message.role === "toolResult") {
			const content = message.content
				.filter((block) => block.type === "text")
				.map((block) => block.text)
				.join("\n");

			messages.push({
				role: "tool",
				content: content || "(no text output)",
				tool_call_id: message.toolCallId,
			});
		}
	}

	return messages;
}

function convertTools(context: Context): ChatTool[] | undefined {
	return context.tools?.map((tool) => ({
		type: "function" as const,
		function: {
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters as Record<string, unknown>,
			strict: false,
		},
	}));
}

function convertResponsesUserContent(content: string | (TextContent | ImageContent)[]): ResponsesInputContent[] {
	if (typeof content === "string") {
		return [{ type: "input_text", text: content }];
	}

	const parts: ResponsesInputContent[] = [];
	for (const part of content) {
		if (part.type === "text") {
			parts.push({ type: "input_text", text: part.text });
		} else if (part.type === "image") {
			parts.push({
				type: "input_image",
				detail: "auto",
				image_url: `data:${part.mimeType};base64,${part.data}`,
			});
		}
	}
	return parts;
}

function convertResponsesMessages(model: Model<Api>, context: Context): ResponsesInput {
	const messages: ResponsesInput = [];

	if (context.systemPrompt) {
		messages.push({
			role: model.reasoning ? "developer" : "system",
			content: [{ type: "input_text", text: context.systemPrompt }],
		});
	}

	let messageIndex = 0;
	for (const message of context.messages) {
		if (message.role === "user") {
			const content = convertResponsesUserContent(message.content);
			if (content.length > 0) {
				messages.push({ role: "user", content });
			}
		} else if (message.role === "assistant") {
			const output: ResponsesInput = [];
			for (const block of message.content) {
				if (block.type === "thinking") {
					const reasoningItem = parseReasoningSignature(block.thinkingSignature);
					if (reasoningItem) output.push(reasoningItem);
				} else if (block.type === "text") {
					const parsedSignature = parseTextSignature(block.textSignature);
					output.push({
						type: "message",
						role: "assistant",
						content: [{ type: "output_text", text: block.text, annotations: [] }],
						status: "completed",
						id: parsedSignature?.id ?? `msg_${messageIndex}`,
						phase: parsedSignature?.phase,
					});
				} else if (block.type === "toolCall") {
					const { callId, itemId } = splitResponsesToolCallId(block.id);
					output.push({
						type: "function_call",
						id: itemId,
						call_id: callId,
						name: block.name,
						arguments: JSON.stringify(block.arguments),
					});
				}
			}
			messages.push(...output);
		} else if (message.role === "toolResult") {
			const textResult = message.content
				.filter((block): block is TextContent => block.type === "text")
				.map((block) => block.text)
				.join("\n");
			const hasImages = message.content.some((block): block is ImageContent => block.type === "image");
			const [callId] = message.toolCallId.split("|");

			let output: string | ResponsesFunctionCallOutputItemList;
			if (hasImages && model.input.includes("image")) {
				const parts: ResponsesFunctionCallOutputItemList = [];
				if (textResult) parts.push({ type: "input_text", text: textResult });
				for (const block of message.content) {
					if (block.type === "image") {
						parts.push({
							type: "input_image",
							detail: "auto",
							image_url: `data:${block.mimeType};base64,${block.data}`,
						});
					}
				}
				output = parts;
			} else {
				output = textResult || (hasImages ? "(see attached image)" : "(no text output)");
			}

			messages.push({
				type: "function_call_output",
				call_id: callId,
				output,
			});
		}
		messageIndex++;
	}

	return messages;
}

function convertResponsesTools(context: Context): ResponsesTool[] | undefined {
	if (!context.tools || context.tools.length === 0) return undefined;

	return context.tools.map((tool) => ({
		type: "function" as const,
		name: tool.name,
		description: tool.description,
		parameters: tool.parameters as Record<string, unknown>,
		strict: false,
	}));
}

function createParams(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): TinfoilAI.Chat.Completions.ChatCompletionCreateParamsStreaming {
	const params: TinfoilAI.Chat.Completions.ChatCompletionCreateParamsStreaming = {
		model: model.id,
		messages: convertMessages(context),
		stream: true as const,
		stream_options: { include_usage: true },
		tools: convertTools(context),
		max_tokens: options?.maxTokens,
		temperature: options?.temperature,
	};

	if (model.id === "gpt-oss-120b" && options?.reasoning) {
		params.reasoning_effort = options.reasoning;
	}

	return params;
}

function createResponsesParams(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): ResponsesCreateParamsStreaming {
	const params: ResponsesCreateParamsStreaming = {
		model: model.id,
		input: convertResponsesMessages(model, context),
		stream: true,
		store: false,
	};
	const tools = convertResponsesTools(context);

	if (options?.maxTokens) {
		params.max_output_tokens = options.maxTokens;
	}

	if (options?.temperature !== undefined) {
		params.temperature = options.temperature;
	}

	if (tools) {
		params.tools = tools;
	}

	if (model.reasoning && options?.reasoning) {
		params.reasoning = {
			effort: options.reasoning,
			summary: "auto",
		};
		params.include = ["reasoning.encrypted_content"];
	}

	return params;
}

export function streamTinfoil(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	(async () => {
		const output = createOutput(model);
		let currentBlock: TextContent | ThinkingContent | StreamingToolCallBlock | null = null;
		const getContentIndex = (block: typeof currentBlock) => (block ? output.content.indexOf(block) : -1);
		const finishCurrentBlock = (block: typeof currentBlock) => {
			if (!block) return;
			const contentIndex = getContentIndex(block);
			if (contentIndex === -1) return;

			if (block.type === "text") {
				stream.push({ type: "text_end", contentIndex, content: block.text, partial: output });
			} else if (block.type === "thinking") {
				stream.push({ type: "thinking_end", contentIndex, content: block.thinking, partial: output });
			} else if (block.type === "toolCall") {
				block.arguments = parseStreamingJson<Record<string, unknown>>(block.partialArgs);
				delete block.partialArgs;
				delete block.streamIndex;
				stream.push({ type: "toolcall_end", contentIndex, toolCall: block, partial: output });
			}
		};

		try {
			const client = getTinfoilClient(options?.apiKey);
			await client.ready();

			const initialParams = createParams(model, context, options);
			const nextParams = await options?.onPayload?.(initialParams, model);
			const params = (nextParams ?? initialParams) as TinfoilAI.Chat.Completions.ChatCompletionCreateParamsStreaming;
			const { data: responseStream, response } = await client.chat.completions
				.create(params, {
					...(options?.signal ? { signal: options.signal } : {}),
					...(options?.headers ? { headers: options.headers } : {}),
					...(options?.timeoutMs !== undefined ? { timeout: options.timeoutMs } : {}),
					...(options?.maxRetries !== undefined ? { maxRetries: options.maxRetries } : {}),
				})
				.withResponse();

			await options?.onResponse?.(
				{
					status: response.status,
					headers: Object.fromEntries(response.headers.entries()),
				},
				model,
			);

			stream.push({ type: "start", partial: output });

			for await (const chunk of responseStream as unknown as AsyncIterable<TinfoilChunk>) {
				output.responseId ||= chunk.id;
				if (chunk.model && chunk.model !== model.id) output.responseModel ||= chunk.model;
				updateUsage(output, chunk.usage, model);

				const choice = chunk.choices?.[0];
				if (!choice) continue;
				updateUsage(output, choice.usage, model);

				if (choice.finish_reason) {
					output.stopReason = mapStopReason(choice.finish_reason);
					if (output.stopReason === "error") {
						output.errorMessage = `Unsupported finish reason: ${choice.finish_reason}`;
					}
				}

				const delta = choice.delta;
				if (!delta) continue;

				if (delta.content) {
					if (!currentBlock || currentBlock.type !== "text") {
						finishCurrentBlock(currentBlock);
						currentBlock = { type: "text", text: "" };
						output.content.push(currentBlock);
						stream.push({ type: "text_start", contentIndex: getContentIndex(currentBlock), partial: output });
					}
					currentBlock.text += delta.content;
					stream.push({
						type: "text_delta",
						contentIndex: getContentIndex(currentBlock),
						delta: delta.content,
						partial: output,
					});
				}

				const reasoning = getReasoningDelta(delta);
				if (reasoning) {
					if (!currentBlock || currentBlock.type !== "thinking") {
						finishCurrentBlock(currentBlock);
						currentBlock = { type: "thinking", thinking: "", thinkingSignature: reasoning.field };
						output.content.push(currentBlock);
						stream.push({ type: "thinking_start", contentIndex: getContentIndex(currentBlock), partial: output });
					}
					currentBlock.thinking += reasoning.text;
					stream.push({
						type: "thinking_delta",
						contentIndex: getContentIndex(currentBlock),
						delta: reasoning.text,
						partial: output,
					});
				}

				for (const toolCall of delta.tool_calls ?? []) {
					const streamIndex = typeof toolCall.index === "number" ? toolCall.index : undefined;
					const sameToolCall =
						currentBlock?.type === "toolCall" &&
						((streamIndex !== undefined && currentBlock.streamIndex === streamIndex) ||
							(streamIndex === undefined && !!toolCall.id && currentBlock.id === toolCall.id));

					if (!sameToolCall) {
						finishCurrentBlock(currentBlock);
						currentBlock = {
							type: "toolCall",
							id: toolCall.id ?? "",
							name: toolCall.function?.name ?? "",
							arguments: {},
							partialArgs: "",
							streamIndex,
						};
						output.content.push(currentBlock);
						stream.push({ type: "toolcall_start", contentIndex: getContentIndex(currentBlock), partial: output });
					}

					const toolBlock = currentBlock?.type === "toolCall" ? currentBlock : undefined;
					if (toolBlock) {
						if (!toolBlock.id && toolCall.id) toolBlock.id = toolCall.id;
						if (!toolBlock.name && toolCall.function?.name) toolBlock.name = toolCall.function.name;
						if (toolBlock.streamIndex === undefined && streamIndex !== undefined) {
							toolBlock.streamIndex = streamIndex;
						}
						const argumentDelta = toolCall.function?.arguments ?? "";
						toolBlock.partialArgs += argumentDelta;
						toolBlock.arguments = parseStreamingJson<Record<string, unknown>>(toolBlock.partialArgs);
						stream.push({
							type: "toolcall_delta",
							contentIndex: getContentIndex(toolBlock),
							delta: argumentDelta,
							partial: output,
						});
					}
				}
			}

			finishCurrentBlock(currentBlock);
			if (options?.signal?.aborted) throw new Error("Request was aborted");
			if (output.stopReason === "aborted") throw new Error("Request was aborted");
			if (output.stopReason === "error") {
				throw new Error(output.errorMessage || "Provider returned an error stop reason");
			}

			stream.push({ type: "done", reason: output.stopReason, message: output });
			stream.end(output);
		} catch (error) {
			removeStreamingScratch(output);
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : String(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end(output);
		}
	})();

	return stream;
}

async function processTinfoilResponsesStream(
	responseStream: AsyncIterable<ResponsesStreamEvent>,
	output: AssistantMessage,
	stream: AssistantMessageEventStream,
	model: Model<Api>,
): Promise<void> {
	let currentItem: ResponsesReasoningItem | ResponsesOutputMessage | ResponsesFunctionToolCall | null = null;
	let currentBlock: ThinkingContent | TextContent | StreamingResponsesToolCallBlock | null = null;
	const blockIndex = () => output.content.length - 1;

	for await (const event of responseStream) {
		if (event.type === "response.created") {
			output.responseId = event.response.id;
			if (event.response.model && event.response.model !== model.id) {
				output.responseModel = event.response.model;
			}
		} else if (event.type === "response.output_item.added") {
			const item = event.item;
			if (item.type === "reasoning") {
				currentItem = item;
				currentBlock = { type: "thinking", thinking: "" };
				output.content.push(currentBlock);
				stream.push({ type: "thinking_start", contentIndex: blockIndex(), partial: output });
			} else if (item.type === "message") {
				currentItem = item;
				currentBlock = { type: "text", text: "" };
				output.content.push(currentBlock);
				stream.push({ type: "text_start", contentIndex: blockIndex(), partial: output });
			} else if (item.type === "function_call") {
				currentItem = item;
				currentBlock = {
					type: "toolCall",
					id: encodeResponsesToolCallId(item.call_id, item.id),
					name: item.name,
					arguments: {},
					partialJson: item.arguments || "",
				};
				output.content.push(currentBlock);
				stream.push({ type: "toolcall_start", contentIndex: blockIndex(), partial: output });
			}
		} else if (event.type === "response.reasoning_summary_part.added") {
			if (currentItem?.type === "reasoning") {
				currentItem.summary.push(event.part);
			}
		} else if (event.type === "response.reasoning_summary_text.delta") {
			if (currentItem?.type === "reasoning" && currentBlock?.type === "thinking") {
				currentBlock.thinking += event.delta;
				const lastPart = currentItem.summary[currentItem.summary.length - 1];
				if (lastPart) lastPart.text += event.delta;
				stream.push({
					type: "thinking_delta",
					contentIndex: blockIndex(),
					delta: event.delta,
					partial: output,
				});
			}
		} else if (event.type === "response.reasoning_summary_part.done") {
			if (currentItem?.type === "reasoning" && currentBlock?.type === "thinking") {
				currentBlock.thinking += "\n\n";
				stream.push({
					type: "thinking_delta",
					contentIndex: blockIndex(),
					delta: "\n\n",
					partial: output,
				});
			}
		} else if (event.type === "response.reasoning_text.delta") {
			if (currentItem?.type === "reasoning" && currentBlock?.type === "thinking") {
				currentBlock.thinking += event.delta;
				stream.push({
					type: "thinking_delta",
					contentIndex: blockIndex(),
					delta: event.delta,
					partial: output,
				});
			}
		} else if (event.type === "response.content_part.added") {
			if (currentItem?.type === "message" && (event.part.type === "output_text" || event.part.type === "refusal")) {
				currentItem.content.push(event.part);
			}
		} else if (event.type === "response.output_text.delta") {
			if (currentBlock?.type === "text") {
				currentBlock.text += event.delta;
				stream.push({
					type: "text_delta",
					contentIndex: blockIndex(),
					delta: event.delta,
					partial: output,
				});
			}
		} else if (event.type === "response.refusal.delta") {
			if (currentBlock?.type === "text") {
				currentBlock.text += event.delta;
				stream.push({
					type: "text_delta",
					contentIndex: blockIndex(),
					delta: event.delta,
					partial: output,
				});
			}
		} else if (event.type === "response.function_call_arguments.delta") {
			if (currentItem?.type === "function_call" && currentBlock?.type === "toolCall") {
				currentBlock.partialJson = `${currentBlock.partialJson ?? ""}${event.delta}`;
				currentBlock.arguments = parseStreamingJson<Record<string, unknown>>(currentBlock.partialJson);
				stream.push({
					type: "toolcall_delta",
					contentIndex: blockIndex(),
					delta: event.delta,
					partial: output,
				});
			}
		} else if (event.type === "response.function_call_arguments.done") {
			if (currentItem?.type === "function_call" && currentBlock?.type === "toolCall") {
				const previousPartialJson = currentBlock.partialJson ?? "";
				currentBlock.partialJson = event.arguments;
				currentBlock.arguments = parseStreamingJson<Record<string, unknown>>(currentBlock.partialJson);

				if (event.arguments.startsWith(previousPartialJson)) {
					const delta = event.arguments.slice(previousPartialJson.length);
					if (delta) {
						stream.push({
							type: "toolcall_delta",
							contentIndex: blockIndex(),
							delta,
							partial: output,
						});
					}
				}
			}
		} else if (event.type === "response.output_item.done") {
			const item = event.item;
			if (item.type === "reasoning" && currentBlock?.type === "thinking") {
				const contentText = item.content?.map((part) => part.text).join("\n\n");
				const summaryText = item.summary.map((part) => part.text).join("\n\n");
				currentBlock.thinking = summaryText || contentText || currentBlock.thinking;
				currentBlock.thinkingSignature = JSON.stringify(item);
				stream.push({
					type: "thinking_end",
					contentIndex: blockIndex(),
					content: currentBlock.thinking,
					partial: output,
				});
				currentBlock = null;
			} else if (item.type === "message" && currentBlock?.type === "text") {
				currentBlock.text =
					item.content.map((part) => (part.type === "output_text" ? part.text : part.refusal)).join("") ||
					currentBlock.text;
				currentBlock.textSignature = encodeTextSignature(item.id, item.phase);
				stream.push({
					type: "text_end",
					contentIndex: blockIndex(),
					content: currentBlock.text,
					partial: output,
				});
				currentBlock = null;
			} else if (item.type === "function_call") {
				const args =
					currentBlock?.type === "toolCall" && currentBlock.partialJson
						? parseStreamingJson<Record<string, unknown>>(currentBlock.partialJson)
						: parseStreamingJson<Record<string, unknown>>(item.arguments || "{}");
				let contentIndex = blockIndex();
				let toolCall: ToolCall;

				if (currentBlock?.type === "toolCall") {
					currentBlock.arguments = args;
					delete currentBlock.partialJson;
					toolCall = currentBlock;
				} else {
					toolCall = {
						type: "toolCall",
						id: encodeResponsesToolCallId(item.call_id, item.id),
						name: item.name,
						arguments: args,
					};
					output.content.push(toolCall);
					contentIndex = blockIndex();
				}

				stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output });
				currentBlock = null;
			}
		} else if (event.type === "response.completed" || event.type === "response.incomplete") {
			const response = event.response;
			output.responseId = response.id;
			if (response.model && response.model !== model.id) {
				output.responseModel = response.model;
			}
			updateResponsesUsage(output, response.usage, model);
			output.stopReason = mapResponsesStopReason(response.status);
			if (output.content.some((block) => block.type === "toolCall") && output.stopReason === "stop") {
				output.stopReason = "toolUse";
			}
		} else if (event.type === "response.failed") {
			const error = event.response.error;
			const details = event.response.incomplete_details;
			const message = error
				? `${error.code || "unknown"}: ${error.message || "no message"}`
				: details?.reason
					? `incomplete: ${details.reason}`
					: "Unknown error (no error details in response)";
			throw new Error(message);
		} else if (event.type === "error") {
			throw new Error(`Error Code ${event.code}: ${event.message}` || "Unknown error");
		}
	}
}

export function streamTinfoilResponses(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	(async () => {
		const output = createOutput(model);

		try {
			const client = getTinfoilClient(options?.apiKey);
			await client.ready();

			const initialParams = createResponsesParams(model, context, options);
			const nextParams = await options?.onPayload?.(initialParams, model);
			const params = (nextParams ?? initialParams) as ResponsesCreateParamsStreaming;
			const { data: responseStream, response } = await client.responses
				.create(params, {
					...(options?.signal ? { signal: options.signal } : {}),
					...(options?.headers ? { headers: options.headers } : {}),
					...(options?.timeoutMs !== undefined ? { timeout: options.timeoutMs } : {}),
					...(options?.maxRetries !== undefined ? { maxRetries: options.maxRetries } : {}),
				})
				.withResponse();

			await options?.onResponse?.(
				{
					status: response.status,
					headers: Object.fromEntries(response.headers.entries()),
				},
				model,
			);

			stream.push({ type: "start", partial: output });
			await processTinfoilResponsesStream(
				responseStream as unknown as AsyncIterable<ResponsesStreamEvent>,
				output,
				stream,
				model,
			);

			if (options?.signal?.aborted) throw new Error("Request was aborted");
			if (output.stopReason === "aborted") throw new Error("Request was aborted");
			if (output.stopReason === "error") {
				throw new Error(output.errorMessage || "Provider returned an error stop reason");
			}

			stream.push({ type: "done", reason: output.stopReason, message: output });
			stream.end(output);
		} catch (error) {
			removeResponsesScratch(output);
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : String(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end(output);
		}
	})();

	return stream;
}

export default function (pi: ExtensionAPI) {
	pi.registerProvider(TINFOIL_PROVIDER, {
		name: "Tinfoil",
		baseUrl: TINFOIL_BASE_URL,
		apiKey: "TINFOIL_API_KEY",
		api: TINFOIL_API,
		models: MODEL_CONFIGS.map(([id, name, contextWindow, multimodal, reasoning, inputCost, outputCost]) => ({
			id,
			name,
			reasoning,
			input: multimodal ? ["text", "image"] : ["text"],
			cost: { input: inputCost, output: outputCost, cacheRead: 0, cacheWrite: 0 },
			contextWindow,
			maxTokens: MAX_TOKENS,
		})),
		streamSimple: streamTinfoil,
	});

	pi.registerProvider(TINFOIL_RESPONSES_PROVIDER, {
		name: "Tinfoil Responses",
		baseUrl: TINFOIL_BASE_URL,
		apiKey: "TINFOIL_API_KEY",
		api: TINFOIL_RESPONSES_API,
		models: RESPONSES_MODEL_CONFIGS.map(
			([id, name, contextWindow, multimodal, reasoning, inputCost, outputCost]) => ({
				id,
				name,
				reasoning,
				input: multimodal ? ["text", "image"] : ["text"],
				cost: { input: inputCost, output: outputCost, cacheRead: 0, cacheWrite: 0 },
				contextWindow,
				maxTokens: MAX_TOKENS,
			}),
		),
		streamSimple: streamTinfoilResponses,
	});
}
