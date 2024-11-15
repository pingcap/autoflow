'use client';

import { isBootstrapStatusPassed } from '@/api/system';
import { useBootstrapStatus } from './BootstrapStatusProvider';

export function SystemWizardBanner () {
  const bootstrapStatus = useBootstrapStatus();
  const configured = isBootstrapStatusPassed(bootstrapStatus);

  if (!configured) {
    return (
      <div className="absolute left-0 top-0 w-full p-1 text-xs text-center bg-yellow-600/10 text-yellow-600 dark:bg-yellow-400/10 dark:text-yellow-400">
        This site is not ready to use yet. Please login or contact admin to finish setup configuration.
      </div>
    );
  }
}
