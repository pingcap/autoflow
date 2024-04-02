import { Flow } from '@/core';
import type { DocumentImportTaskProcessor } from '@/core/services/importing';
import type { CreateDocument } from '@/core/v1/document';
import type { CreateDocumentImportTask } from '@/core/v1/document_import_task';
import { md5 } from '@/lib/digest';
import { genId } from '@/lib/id';

export function createDocumentImportTaskProcessor (flow: Flow): DocumentImportTaskProcessor {
  return async (task) => {
    const processor = flow.getImportSourceTaskProcessor(task.type, task.url);
    const { enqueue, content } = await processor.process(task);

    let tasks: CreateDocumentImportTask[] = [];
    let document: CreateDocument | undefined;

    if (enqueue) {
      tasks = enqueue.map(info => ({
        ...info,
        status: 'CREATED',
        source_id: task.source_id,
        created_at: new Date(),
      }));
    }

    if (content) {
      const storage = flow.getStorage();
      const uri = await storage.put(`raw-documents/${genId()}`, content.buffer);
      document = {
        content_uri: uri,
        source_uri: task.url,
        mime: content.mime,
        name: '',
        created_at: new Date(),
        last_modified_at: new Date(),
        hash: md5(content.buffer),
      };
    }

    return {
      tasks,
      document,
    };
  };
}