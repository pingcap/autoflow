import { z } from 'zod';

export const reCaptchas = z.enum(['', 'v3', 'enterprise']);

export const SecuritySetting = z.object({
  google_recaptcha: reCaptchas,
  google_recaptcha_site_key: z.string().optional(),
  google_recaptcha_secret_key: z.string().optional(),
  google_recaptcha_enterprise_project_id: z.string().optional(),
});

export const SecuritySettingResult = SecuritySetting.partial();
export type ISecuritySettingResult = z.infer<typeof SecuritySettingResult>;
