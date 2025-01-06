import { authenticationHeaders, handleErrors, handleResponse, requestUrl } from '@/lib/request';
import { z } from 'zod';

interface SettingItemBase<K, T> {
  name: string;
  description: string;
  group: string;
  data_type: K;
  value: T | null;
  client: boolean | null;
  default: T;
}

export type IntSettingItem = SettingItemBase<'int', number>
export type FloatSettingItem = SettingItemBase<'float', number>
export type BoolSettingItem = SettingItemBase<'bool', boolean>
export type StringSettingItem = SettingItemBase<'str', string>
export type ListSettingItem = SettingItemBase<'list', any[]>
export type DictSettingItem = SettingItemBase<'dict', object>

export type SettingItem =
  IntSettingItem
  | FloatSettingItem
  | BoolSettingItem
  | StringSettingItem
  | ListSettingItem
  | DictSettingItem;

export interface PublicWebsiteSettings {
  'title': string;
  'description': string;
  'homepage_title': string;
  'homepage_example_questions': string[];
  'homepage_footer_links': { text: string, href: string }[];
  'logo_in_dark_mode': string;
  'logo_in_light_mode': string;
  'social_github': string | null;
  'social_twitter': string | null;
  'social_discord': string | null;
  'custom_js_example_questions': string[];
  'custom_js_button_label': string;
  'custom_js_button_img_src': string;
  'custom_js_logo_src': string;
  'ga_id': string | null;
  'max_upload_file_size': number | null;
  'enable_post_verifications': boolean;
  'enable_post_verifications_for_widgets': boolean;
}

const settingsItemSchema = z.object({
  name: z.string(),
  description: z.string(),
  client: z.boolean().nullable(),
  group: z.string(),
  // data_type: z.enum(['list', 'dict', 'str', 'int', 'float', 'bool']),
}).and(z.discriminatedUnion('data_type', [
  z.object({
    data_type: z.literal('str'),
    value: z.string().nullable(),
    default: z.string(),
  }),
  z.object({
    data_type: z.literal('int'),
    value: z.number().int().nullable(),
    default: z.number().int(),
  }),
  z.object({
    data_type: z.literal('float'),
    value: z.number().nullable(),
    default: z.number(),
  }),
  z.object({
    data_type: z.literal('bool'),
    value: z.boolean().nullable(),
    default: z.coerce.boolean(),
  }),
  z.object({
    data_type: z.literal('list'),
    value: z.any().array().nullable(),
    default: z.any().array(),
  }),
  z.object({
    data_type: z.literal('dict'),
    value: z.object({}).passthrough().nullable(),
    default: z.object({}).passthrough(),
  }),
]));

export type AllSettings = Record<string, SettingItem>

export async function getAllSiteSettings (): Promise<AllSettings> {
  return await fetch(requestUrl(`/api/v1/admin/site-settings`),
    {
      headers: await authenticationHeaders(),
    })
    .then(handleResponse(z.record(settingsItemSchema)));
}

export async function updateSiteSetting (name: string, value: any) {
  await fetch(requestUrl(`/api/v1/admin/site-settings/${name}`), {
    method: 'PUT',
    headers: {
      ...await authenticationHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ value }),
  }).then(handleErrors);
}

export async function getPublicSiteSettings (): Promise<PublicWebsiteSettings> {
  return fetch(requestUrl(`/api/v1/site-config`), {
    headers: await authenticationHeaders(),
    credentials: 'include',
  }).then(handleErrors).then(res => res.json());
}
