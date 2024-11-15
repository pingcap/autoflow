import { getKnowledgeBaseById } from '@/api/knowledge-base';
import { cache } from 'react';

export const cachedGetKnowledgeBaseById = cache(getKnowledgeBaseById);