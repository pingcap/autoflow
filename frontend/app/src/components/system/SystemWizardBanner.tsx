'use client';

import { isBootstrapStatusPassed } from '@/api/system';
import { useBootstrapStatus } from './BootstrapStatusProvider';

export function SystemWizardBanner () {
  const bootstrapStatus = useBootstrapStatus();
  const configured = isBootstrapStatusPassed(bootstrapStatus);

  if (!configured) {
    return (
      <div className="absolute left-0 top-0 w-full p-1 text-xs text-center bg-warning/10 text-warning">
        This site is not ready to use yet. Please login or contact admin to finish setup configuration.
      </div>
    );
  }
}
