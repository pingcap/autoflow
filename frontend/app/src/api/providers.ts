import { z, type ZodType } from 'zod';

export interface ProviderOption {
  provider: string;
  provider_display_name: string | null;
  provider_description: string | null;
  provider_url: string | null;
  credentials_display_name: string;
  credentials_description: string;
  credentials_type: 'str' | 'dict';
  default_credentials: any;
  default_config: object;
  config_description: string;
}

export const providerOptionSchema = z.object({
  provider: z.string(),
  provider_display_name: z.string().nullable(),
  provider_description: z.string().nullable(),
  provider_url: z.string().nullable(),
  credentials_display_name: z.string(),
  credentials_description: z.string(),
  default_config: z.object({}).passthrough(),
  config_description: z.string(),
}).and(z.discriminatedUnion('credentials_type', [
  z.object({
    credentials_type: z.literal('str'),
    default_credentials: z.string(),
  }),
  z.object({
    credentials_type: z.literal('dict'),
    default_credentials: z.object({}).passthrough(),
  }),
])) satisfies ZodType<ProviderOption, any>;
