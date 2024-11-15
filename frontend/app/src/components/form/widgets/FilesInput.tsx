import { DataTable } from '@/components/data-table';
import { DataTableHeading } from '@/components/data-table-heading';
import type { FormControlWidgetProps } from '@/components/form/control-widget';
import { Button } from '@/components/ui/button';
import type { ColumnDef } from '@tanstack/react-table';
import { createColumnHelper } from '@tanstack/table-core';
import { filesize } from 'filesize';
import { FileMinus2Icon } from 'lucide-react';
import { type ChangeEvent, forwardRef, useId } from 'react';

export interface FilesInputProps extends FormControlWidgetProps {
  accept: string[];
}

const helper = createColumnHelper<File>();

export const FilesInput = forwardRef<any, FilesInputProps>(({
  accept,
  name,
  id,
  disabled,
  onBlur,
  value: files = [],
  onChange: onFilesChange,
  ...props
}, ref) => {
  const hookId = useId();
  id = id ?? hookId;

  const columns: ColumnDef<File, any>[] = [
    helper.accessor('name', {}),
    helper.accessor('type', {}),
    helper.accessor('size', { cell: cell => filesize(cell.getValue()) }),
    helper.display({
      id: 'op',
      cell: (cell) => <Button
        type="button"
        variant="ghost"
        size="sm"
        className='text-xs'
        onClick={() => {
          files = [...files];
          files.splice(cell.row.index, 1);
          onFilesChange?.(files);
        }}
      >
        <FileMinus2Icon className="size-4 mr-1" />
        Remove
      </Button>,
    }),
  ];

  const handleSelectFiles = (ev: ChangeEvent<HTMLInputElement>) => {
    ev.preventDefault();
    if (ev.target.files) {
      const newFiles = Array.from(ev.target.files);
      onFilesChange?.([...files, ...newFiles]);
    }
  };

  return (
    <>
      <DataTable<File, any>
        classNames={{
          td: 'px-2 py-1',
        }}
        before={
          <DataTableHeading>
            <input
              className="hidden"
              id={id}
              name={name}
              type="file"
              multiple
              accept={accept.join(', ')}
              onChange={handleSelectFiles}
              disabled={disabled}
            />
            <Button
              variant="secondary"
              disabled={disabled}
              ref={ref}
              onBlur={onBlur}
              {...props}
              onClick={(event) => {
                (props as any).onClick?.(event);
                if (!event.defaultPrevented) {
                  document.getElementById(id)?.click();
                }
              }}
              type="button"
            >
              Select files...
            </Button>
          </DataTableHeading>
        }
        columns={columns}
        data={files}
        hideHeader
      />
    </>
  );
});

FilesInput.displayName = 'FilesInput';
