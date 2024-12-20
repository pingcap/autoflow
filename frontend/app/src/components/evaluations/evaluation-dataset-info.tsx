import { type EvaluationDataset, updateEvaluationDataset } from '@/api/evaluations';
import { mutateEvaluationDatasets, useEvaluationDataset } from '@/components/evaluations/hooks';
import { FormInput } from '@/components/form/control-widget';
import { formFieldLayout } from '@/components/form/field-layout';
import { createAccessorHelper, GeneralSettingsFieldBeta as GeneralSettingsField, GeneralSettingsForm } from '@/components/settings-form';
import { Skeleton } from '@/components/ui/skeleton';
import { useRouter } from 'next/navigation';
import * as React from 'react';
import { useTransition } from 'react';
import { z } from 'zod';

export function EvaluationDatasetInfo ({ evaluationDatasetId }: { evaluationDatasetId: number }) {
  const { evaluationDataset } = useEvaluationDataset(evaluationDatasetId);

  if (evaluationDataset) {
    return <EvaluationDatasetInfoDisplay evaluationDataset={evaluationDataset} />;
  } else {
    return <EvaluationDatasetInfoSkeleton />;
  }
}

const field = formFieldLayout<Record<'value', any>>();

export function EvaluationDatasetInfoDisplay ({ evaluationDataset }: { evaluationDataset: EvaluationDataset }) {
  const router = useRouter();
  const [transitioning, startTransition] = useTransition();

  return (
    <div className="space-y-4 max-w-screen-sm">
      <GeneralSettingsForm
        data={evaluationDataset}
        readonly={transitioning}
        loading={transitioning}
        onUpdate={async (item) => {
          await updateEvaluationDataset(item.id, { name: item.name });
          startTransition(() => {
            router.refresh();
            void mutateEvaluationDatasets();
          });
        }}
      >
        <GeneralSettingsField
          accessor={id}
          schema={whateverSchema}
          readonly
        >
          <field.Basic name="value" label="ID">
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField
          accessor={name}
          schema={nameSchema}
        >
          <field.Basic name="value" label="Name">
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField
          accessor={createdAt}
          schema={whateverSchema}
          readonly
        >
          <field.Basic name="value" label="Created At">
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField
          accessor={updatedAt}
          schema={whateverSchema}
          readonly
        >
          <field.Basic name="value" label="Updated At">
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
        <GeneralSettingsField
          accessor={userId}
          schema={whateverSchema}
          readonly
        >
          <field.Basic name="value" label="User ID">
            <FormInput />
          </field.Basic>
        </GeneralSettingsField>
      </GeneralSettingsForm>
    </div>
  );
}

export function EvaluationDatasetInfoSkeleton ({}: {}) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="py-[0.125em] text-xs">
          <Skeleton className="block w-[14em] h-[1em] rounded-sm" />
        </div>
        <div className="py-[0.125em] text-xs">
          <Skeleton className="block w-[14em] h-[1em] rounded-sm" />
        </div>
        <div className="py-[0.125em] text-xs">
          <Skeleton className="block w-[8em] h-[1em] rounded-sm" />
        </div>
      </div>
    </div>
  );
}

const helper = createAccessorHelper<EvaluationDataset>();

const id = helper.field('id');
const name = helper.field('name');
const userId = helper.field('user_id');
const createdAt = helper.dateField('created_at');
const updatedAt = helper.dateField('updated_at');

const whateverSchema = z.any();
const nameSchema = z.string().min(1);