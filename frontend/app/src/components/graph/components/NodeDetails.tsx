import { getEntity } from '@/api/graph';
import { Loader } from '@/components/loader';
import { toastError, toastSuccess } from '@/lib/ui-error';
import { cn } from '@/lib/utils';
import { useContext, useEffect, useMemo, useState } from 'react';
import type { IdType } from '../network/Network';
import { useRemote } from '../remote';
import { useDirtyEntity } from '../useDirtyEntity';
import { type Entity, handleServerEntity } from '../utils';
import { EditingButton } from './EditingButton';
import { InputField } from './InputField';
import { JsonField } from './JsonField';
import { NetworkContext } from './NetworkContext';
import { TextareaField } from './TextareaField';

const loadEntity = (kbId: number, id: number) => getEntity(kbId, id).then(handleServerEntity);

export function NodeDetails ({
  knowledgeBaseId,
  entity,
  onClickTarget,
  onUpdate,
  onEnterSubgraph,
}: {
  knowledgeBaseId: number,
  entity: Entity,
  onClickTarget?: (target: { type: string, id: IdType }) => void;
  onUpdate?: (newData: Entity) => void
  onEnterSubgraph: (type: string, entityId: IdType) => void
}) {
  const [editing, setEditing] = useState(false);
  const network = useContext(NetworkContext);

  const neighbors = useMemo(() => {
    return Array.from(network.nodeNeighborhoods(entity.id) ?? []).map(id => network.node(id)!);
  }, [network, entity.id]);

  const latestData = useRemote(entity, loadEntity, knowledgeBaseId, Number(entity.id));
  const dirtyEntity = useDirtyEntity(knowledgeBaseId, entity.id);

  // dirty set
  entity = latestData.data;

  const handleSave = () => {
    void dirtyEntity.save()
      .then((newEntityData) => {
        setEditing(false);
        onUpdate?.(latestData.mutate(prev => Object.assign({}, prev, newEntityData)));
        toastSuccess('Successfully saved.');
      })
      .catch((error: any) => {
        toastError('Failed to save entity', error);
      });
  };

  const handleReset = () => {
    dirtyEntity.resetSave();
    dirtyEntity.reset(entity);
    setEditing(false);
  };

  useEffect(() => {
    handleReset();
    onUpdate?.(latestData.data);
  }, [latestData.data]);

  const busy = dirtyEntity.saving || latestData.revalidating;
  const controlsDisabled = !editing || busy;

  return (
    <div className="p-4 space-y-4 h-full overflow-y-auto">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground font-normal ">
          <b>#{entity.id}</b> {entity.entity_type} entity
        </span>
        <EditingButton onEnterSubgraph={() => onEnterSubgraph('entity', entity.id)} editing={editing} onStartEdit={() => setEditing(true)} onSave={handleSave} onReset={handleReset} busy={busy} />
      </div>
      {entity.synopsis_info?.topic && <section>
        <h6 className="text-xs font-bold text-accent-foreground mb-1">Synopsis topic</h6>
        <p className="block w-full text-xs text-accent-foreground">
          {entity.synopsis_info.topic}
        </p>
      </section>}
      <InputField label="Name" ref={dirtyEntity.nameRef} defaultValue={entity.name} disabled={controlsDisabled} />
      <TextareaField label="Description" ref={dirtyEntity.descriptionRef} defaultValue={entity.description} disabled={controlsDisabled} />
      <JsonField label="Meta" ref={dirtyEntity.metaRef} defaultValue={entity.meta} disabled={controlsDisabled} />
      <section>
        <h6 className="text-xs font-bold text-accent-foreground mb-1">Neighborhoods</h6>
        <ul className={cn('w-full max-h-40 overflow-y-auto bg-card rounded border transition-opacity', editing && 'opacity-50 pointer-events-none')}>
          {neighbors.map(entity => (
            <li
              key={entity.id}
              className={'text-xs p-1 border-b last-of-type:border-b-0 cursor-pointer hover:text-primary hover:bg-primary/10 transition-colors'}
              onClick={() => {
                if (!editing) {
                  onClickTarget?.({ type: 'node', id: entity.id });
                }
              }}
            >
              {entity.name}
              <span className="text-muted-foreground">
                {' '}#{entity.id}
              </span>
            </li>
          ))}
        </ul>
      </section>
      <Loader loading={latestData.revalidating}>
        Loading entity #{entity.id}
      </Loader>
    </div>
  );
}