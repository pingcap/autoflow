import type { CellContext } from '@tanstack/react-table';
import Link from 'next/link';

export interface LinkCellProps<Row> {
  icon?: React.ReactNode;
  url?: (row: Row) => string;
  text?: (row: Row) => string;
  truncate?: boolean;
  truncate_length?: number;
}

const format_link = (url: string, maxLength: number = 30): string => {
  if (!url || url.length <= maxLength) return url;
  const start = url.substring(0, maxLength / 2);
  const end = url.substring(url.length - maxLength / 2);
  return `${start}...${end}`;
};

export function link<Row> ({ icon, url, text, truncate, truncate_length }: LinkCellProps<Row>) {
  // eslint-disable-next-line react/display-name
  return (context: CellContext<Row, any>) => {
    const href_value = url ? url(context.row.original) : String(context.getValue());
    const text_value = text ? text(context.row.original) : String(context.getValue());
    const display_text = truncate ? format_link(text_value, truncate_length) : text_value;

    return <Link
      className="underline font-mono flex items-center gap-1"
      href={href_value}>
      {icon} {display_text}
    </Link>
  };
}
