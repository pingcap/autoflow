import { updateEvaluationDatasetItem, type UpdateEvaluationDatasetItemParams } from '@/api/evaluations';
import { mutateEvaluationDataset, useEvaluationDatasetItem } from '@/components/evaluations/hooks';
import { FormTextarea } from '@/components/form/control-widget';
import { formFieldLayout } from '@/components/form/field-layout';
import { CodeInput } from '@/components/form/widgets/CodeInput';
import { createAccessorHelper, GeneralSettingsForm } from '@/components/settings-form';
import { GeneralSettingsField } from '@/components/settings-form/GeneralSettingsField';
import { z } from 'zod';

const field = formFieldLayout<{ value: any }>();

export function UpdateEvaluationDatasetItemForm ({ evaluationDatasetId, evaluationDatasetItemId }: { evaluationDatasetId: number, evaluationDatasetItemId: number }) {
  const {
    evaluationDatasetItem,
    isLoading,
    isValidating,
    mutate,
  } = useEvaluationDatasetItem(evaluationDatasetId, evaluationDatasetItemId);

  if (!evaluationDatasetItem) {
    return <></>;
  }

  return (
    <div className="space-y-4 max-w-screen-sm">
      <GeneralSettingsForm
        readonly={false}
        data={evaluationDatasetItem}
        loading={!evaluationDatasetItem || isLoading || isValidating}
        onUpdate={async ({ query, reference, retrieved_contexts, extra }) => {
          const item = await updateEvaluationDatasetItem(
            evaluationDatasetId,
            evaluationDatasetItemId,
            {
              query, retrieved_contexts, reference, extra,
            },
          );
          void mutate(item, { revalidate: true });
          void mutateEvaluationDataset(evaluationDatasetId);
        }}
      >
        <GeneralSettingsField accessor={query} schema={textSchema}>
          <field.Basic name="value" label="Query">
            <CodeInput language="markdown" />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField accessor={reference} schema={textSchema}>
          <field.Basic name="value" label="Reference">
            <CodeInput language="markdown" />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField accessor={retrievedContexts} schema={textListSchema}>
          <field.PrimitiveArray name="value" label="Retrieved Contexts" newItemValue={() => ''}>
            <FormTextarea />
          </field.PrimitiveArray>
        </GeneralSettingsField>
        <GeneralSettingsField accessor={extra} schema={jsonSchema}>
          <field.Basic name="value" label="Extra">
            <CodeInput language="json" />
          </field.Basic>
        </GeneralSettingsField>
      </GeneralSettingsForm>
    </div>
  );
}

const helper = createAccessorHelper<UpdateEvaluationDatasetItemParams>();

const query = helper.field('query');
const reference = helper.field('reference');
const retrievedContexts = helper.field('retrieved_contexts');
const extra = helper.jsonTextField('extra');

const textSchema = z.string().min(1);
const textListSchema = z.string().min(1, 'Non empty').array();
const jsonSchema = z.any();
