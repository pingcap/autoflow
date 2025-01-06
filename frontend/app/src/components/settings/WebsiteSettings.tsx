'use client';

import type { AllSettings } from '@/api/site-settings';
import { LinkArrayField } from '@/components/settings/LinkArrayField';
import { SettingsField } from '@/components/settings/SettingsField';
import { StringArrayField } from '@/components/settings/StringArrayField';
import { Separator } from '@/components/ui/separator';
import { z } from 'zod';

export function WebsiteSettings ({ schema }: { schema: AllSettings }) {
  return (
    <div className="space-y-8 max-w-screen-md">
      <section className="space-y-6">
        <h2 className="text-lg font-medium">Basic Settings</h2>
        <SettingsField name="title" item={schema.title} />
        <SettingsField name="description" item={schema.description} />
        <SettingsField name="logo_in_dark_mode" item={schema.logo_in_dark_mode} />
        <SettingsField name="logo_in_light_mode" item={schema.logo_in_light_mode} />
      </section>
      <Separator />
      <section className="space-y-6">
        <h2 className="text-lg font-medium">Homepage Settings</h2>
        <SettingsField name="homepage_title" item={schema.homepage_title} />
        <SettingsField name="homepage_example_questions" item={schema.homepage_example_questions} arrayItemSchema={z.string()}>
          {props => <StringArrayField {...props} />}
        </SettingsField>
        <SettingsField name="homepage_footer_links" item={schema.homepage_footer_links} arrayItemSchema={z.object({ text: z.string(), href: z.string() })}>
          {props => <LinkArrayField {...props} />}
        </SettingsField>
      </section>
      <Separator />
      <section className="space-y-6">
        <h2 className="text-lg font-medium">Social links</h2>
        <SettingsField name="social_github" item={schema.social_github} />
        <SettingsField name="social_twitter" item={schema.social_twitter} />
        <SettingsField name="social_discord" item={schema.social_discord} />
      </section>
      <Separator />
      <section className="space-y-6">
        <h2 className="text-lg font-medium">Analytics</h2>
        <SettingsField name="ga_id" item={schema.ga_id} />
      </section>
      <Separator />
      <section className="space-y-6">
        <h2 className="text-lg font-medium">Uploads</h2>
        <SettingsField name="max_upload_file_size" item={schema.max_upload_file_size} />
      </section>
    </div>
  );
}