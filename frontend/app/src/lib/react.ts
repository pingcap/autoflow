import type { ChangeEvent, SyntheticEvent } from 'react';

export function trigger<T extends typeof HTMLTextAreaElement | typeof HTMLInputElement> (inputElement: InstanceType<T>, Element: T, value: string) {
  // https://stackoverflow.com/questions/23892547/what-is-the-best-way-to-trigger-change-or-input-event-in-react-js
  const set = Object.getOwnPropertyDescriptor(Element.prototype, 'value')!.set!;
  set.call(inputElement, value);
  const event = new Event('input', { bubbles: true });
  inputElement.dispatchEvent(event);
}

export function isEvent (value: unknown): value is SyntheticEvent {
  if (!value) {
    return false;
  }

  if (typeof value !== 'object') {
    return false;
  }

  for (const name of ['stopPropagation', 'preventDefault', 'type']) {
    if (!(name in value)) {
      return false;
    }
  }

  return true;
}

export function isChangeEvent (value: unknown): value is ChangeEvent {
  return isEvent(value) && value.type === 'change';
}
