import { type KnowledgeBase, type KnowledgeBaseChunkingConfig, type KnowledgeBaseChunkingConfigAdvanced, type KnowledgeBaseChunkingConfigGeneral, knowledgeBaseChunkingConfigSchema, type KnowledgeBaseChunkingMarkdownSplitterConfig, type KnowledgeBaseChunkingSentenceSplitterConfig, type KnowledgeBaseChunkingSplitterRule } from '@/api/knowledge-base';
import { FormInput } from '@/components/form/control-widget';
import { formFieldLayout } from '@/components/form/field-layout';
import { createAccessorHelper, GeneralSettingsField } from '@/components/settings-form';
import { FormField, FormItem, FormLabel } from '@/components/ui/form.beta';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';

const helper = createAccessorHelper<KnowledgeBase>();

export function KnowledgeBaseChunkingConfigFields () {
  return (
    <GeneralSettingsField
      accessor={helper.field('chunking_config', defaultConfig)}
      schema={knowledgeBaseChunkingConfigSchema}
    >
      <ModeSwitch />
    </GeneralSettingsField>
  );
}

const fieldLayout = formFieldLayout<{ value: KnowledgeBaseChunkingConfigGeneral }>();
const advancedFieldLayout = formFieldLayout<{ value: KnowledgeBaseChunkingConfigAdvanced }>();

function ModeSwitch () {
  return (
    <FormField<{ value: KnowledgeBase['chunking_config'] }, 'value'>
      name="value"
      render={(field, form) => <>
        <FormItem>
          <FormLabel>
            Chunking Mode
          </FormLabel>
          <ToggleGroup
            className="w-full flex items-center"
            type="single"
            value={field.state.value?.mode ?? undefined}
            onValueChange={(value => {
              field.setValue(switchMode(value as never));
            })}
            onBlur={field.handleBlur}
          >
            <ToggleGroupItem className="flex-1 border block text-left font-normal h-auto py-4 opacity-50 data-[state=on]:opacity-100 hover:opacity-100 hover:bg-transparent hover:text-foreground transition-all" value="general">
              <div className="font-semibold">
                General
              </div>
              <p className="text-muted-foreground text-xs">[TODO]</p>
            </ToggleGroupItem>
            <ToggleGroupItem className="flex-1 border block text-left font-normal h-auto py-4 opacity-50 data-[state=on]:opacity-100 hover:opacity-100 hover:bg-transparent hover:text-foreground transition-all" value="advanced">
              <div className="font-semibold">
                Advanced
              </div>
              <p className="text-muted-foreground text-xs">[TODO]</p>
            </ToggleGroupItem>
          </ToggleGroup>
          <div className="pl-4 border-l-4">
            {form.state.values.value?.mode === 'general' && <GeneralChunkingConfig />}
            {form.state.values.value?.mode === 'advanced' && <AdvancedChunkingConfig />}
          </div>
        </FormItem>
      </>}
    />
  );
}

function GeneralChunkingConfig () {
  return (
    <div className="grid md:grid-cols-3 gap-4">
      <fieldLayout.Basic name="value.chunk_size" label="Chunk Size">
        <FormInput type="number" />
      </fieldLayout.Basic>
      <fieldLayout.Basic name="value.chunk_overlap" label="Chunk Overlap">
        <FormInput type="number" />
      </fieldLayout.Basic>
      <fieldLayout.Basic name="value.paragraph_separator" label="Paragraph Separator">
        <FormInput />
      </fieldLayout.Basic>
    </div>
  );
}

function AdvancedChunkingConfig () {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="text-sm font-medium text-muted-foreground">Plain Text (text/plain)</div>
        <SplitterRuleConfig rule="text/plain" />
      </div>
      <div className="space-y-2">
        <div className="text-sm font-medium text-muted-foreground">Markdown (text/markdown)</div>
        <SplitterRuleConfig rule="text/markdown" />
      </div>
    </div>
  );
}

function SplitterRuleConfig ({ rule }: { rule: keyof KnowledgeBaseChunkingConfigAdvanced['rules'] }) {
  const name = `value.rules.${rule}` as const;
  return (
    <div className="space-y-4">
      <FormField<{ value: KnowledgeBaseChunkingConfigAdvanced }, typeof name>
        name={name}
        render={(field, form) => (
          <>
            <Select
              name={name}
              value={field.state.value.splitter}
              onValueChange={value => {
                field.setValue(({
                  splitter: value,
                  splitter_config: switchSplitter(value as never),
                } as KnowledgeBaseChunkingSplitterRule));
              }}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="SentenceSplitter">SentenceSplitter</SelectItem>
                <SelectItem value="MarkdownSplitter">MarkdownSplitter</SelectItem>
              </SelectContent>
            </Select>

            {field.state.value.splitter === 'SentenceSplitter' && (
              <div className="pl-4 grid grid-cols-3 gap-4">
                <advancedFieldLayout.Basic name={`value.rules.${rule}.splitter_config.chunk_size`} label="Chunk Size">
                  <FormInput type="number" min={1} step={1} />
                </advancedFieldLayout.Basic>
                <advancedFieldLayout.Basic name={`value.rules.${rule}.splitter_config.chunk_overlap`} label="Chunk Overlap">
                  <FormInput type="number" min={0} step={1} />
                </advancedFieldLayout.Basic>
                <advancedFieldLayout.Basic name={`value.rules.${rule}.splitter_config.paragraph_separator`} label="Paragraph Separator">
                  <FormInput />
                </advancedFieldLayout.Basic>
              </div>
            )}
            {field.state.value.splitter === 'MarkdownSplitter' && (
              <div className="pl-4 grid grid-cols-3 gap-4">
                <advancedFieldLayout.Basic name={`value.rules.${rule}.splitter_config.chunk_size`} label="Chunk Size">
                  <FormInput type="number" min={1} step={1} />
                </advancedFieldLayout.Basic>
                <advancedFieldLayout.Basic name={`value.rules.${rule}.splitter_config.chunk_header_level`} label="Chunk Header Level">
                  <FormInput type="number" min={1} max={6} step={1} />
                </advancedFieldLayout.Basic>
              </div>
            )}
          </>
        )}
      />
    </div>
  );
}

function switchMode (mode: KnowledgeBaseChunkingConfig['mode']): KnowledgeBaseChunkingConfig {
  switch (mode) {
    case 'general':
      return {
        mode: 'general',
        ...switchSplitter('SentenceSplitter'),
      };
    case 'advanced': {
      return {
        mode: 'advanced',
        rules: {
          'text/plain': {
            splitter: 'SentenceSplitter',
            splitter_config: switchSplitter('SentenceSplitter'),
          },
          'text/markdown': {
            splitter: 'MarkdownSplitter',
            splitter_config: switchSplitter('MarkdownSplitter'),
          },
        },
      };
    }
  }
}

function switchSplitter (splitter: 'SentenceSplitter'): KnowledgeBaseChunkingSentenceSplitterConfig;
function switchSplitter (splitter: 'MarkdownSplitter'): KnowledgeBaseChunkingMarkdownSplitterConfig;
function switchSplitter (splitter: 'SentenceSplitter' | 'MarkdownSplitter') {
  switch (splitter) {
    case 'SentenceSplitter':
      return {
        chunk_size: 200,
        chunk_overlap: 200,
        paragraph_separator: '\\n\\n',
      } satisfies KnowledgeBaseChunkingSentenceSplitterConfig;
    case 'MarkdownSplitter':
      return {
        chunk_size: 200,
        chunk_header_level: 3,
      } satisfies KnowledgeBaseChunkingMarkdownSplitterConfig;
  }
}

const defaultConfig = switchMode('general');
