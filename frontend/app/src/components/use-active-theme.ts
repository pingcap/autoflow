import { useTheme } from 'next-themes';

/**
 * Respect 'system' and returns current style. Default to 'light'.
 */
export function useActiveTheme () {
  const { theme, systemTheme } = useTheme();
  const selectedTheme = theme === 'system' ? systemTheme : theme;

  return selectedTheme === 'dark' ? 'dark' : 'light';
}

export type AvailableTheme = ReturnType<typeof useActiveTheme>;