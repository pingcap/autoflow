'use client';

import { createContext, type Dispatch, type SetStateAction, useContext } from 'react';

export const ManagedPanelContext = createContext<{ open: boolean, setOpen: Dispatch<SetStateAction<boolean>> }>({
  open: false,
  setOpen: () => {},
});

export function useManagedPanel () {
  return useContext(ManagedPanelContext);
}
