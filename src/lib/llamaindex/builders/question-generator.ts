import {observeQuestionGenerator} from "@/lib/langfuse/llamaindex/question-generator";
import {QuestionGeneratorConfig, QuestionGeneratorProvider} from "@/lib/llamaindex/config/question-generator";
import {promptParser} from "@/lib/llamaindex/prompts/PromptParser";
import {LangfuseTraceClient} from "langfuse";
import {defaultSubQuestionPrompt, LLMQuestionGenerator} from "llamaindex";
import type {BaseQuestionGenerator} from "llamaindex/engines/query/types";
import {BaseLLM} from "llamaindex/llm/base";

export const DEFAULT_QUESTION_GENERATOR_CONFIG: QuestionGeneratorConfig = {
  provider: QuestionGeneratorProvider.LLM,
  options: {}
};

export function buildQuestionGenerator (
  config: QuestionGeneratorConfig = DEFAULT_QUESTION_GENERATOR_CONFIG,
  promptContext: Record<string, any>,
  llm?: BaseLLM,
  trace?: LangfuseTraceClient,
): BaseQuestionGenerator {
  switch (config.provider) {
    case QuestionGeneratorProvider.LLM:
      const generator = new LLMQuestionGenerator({
        prompt: promptParser.getPrompt(config.options?.prompt, defaultSubQuestionPrompt, promptContext),
        llm
      });
      return observeQuestionGenerator(generator, trace);
    default:
      throw new Error(`Unknown question generator provider: ${config}`)
  }
}