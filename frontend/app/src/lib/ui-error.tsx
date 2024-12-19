import { toast } from 'sonner';

export function toastSuccess (message: string) {
  toast.success(message);
}

export function toastError (title: string, error: unknown) {
  toast.error(title, { description: getErrorMessage(error) });
}

function getErrorMessage (error: unknown) {
  if (!error) {
    return 'Unknown error message';
  }
  if (typeof error === 'object') {
    if ('message' in error) {
      return String(error.message);
    }
  }
  return String(error);
}
