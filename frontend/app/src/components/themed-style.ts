'use client';

import { type AvailableTheme, useActiveTheme } from '@/components/use-active-theme';
import { cloneElement, type CSSProperties, type ReactElement } from 'react';

export function ThemedStyle ({ children, ...themes }: { children: ReactElement<{ style?: CSSProperties }> } & Record<AvailableTheme, CSSProperties>) {
  const theme = useActiveTheme();

  return cloneElement(children, {
    style: {
      ...themes[theme],
      ...children.props.style,
    },
  });
}
