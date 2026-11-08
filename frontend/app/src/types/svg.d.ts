declare module '*.svg' {
  import type { FC, SVGProps } from 'react';
  const SvgComponent: FC<SVGProps<SVGSVGElement>>;
  export default SvgComponent;
}
