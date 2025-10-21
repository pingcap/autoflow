import type { SimulationNodeDatum } from 'd3';

export interface NetworkRendererOptions<Node, Link> {
  showId?: boolean;
  showLinkLabel?: boolean;

  getNodeInitialAttrs?: (node: Node, index: number) => Pick<SimulationNodeDatum, 'x' | 'y'>;

  getNodeLabel?: (node: Node) => string | undefined;
  getNodeDetails?: (node: Node) => string | undefined;
  getNodeMeta?: (node: Node) => any;
  getNodeRadius?: (node: Node) => number;
  getNodeColor?: (node: Node) => string;
  getNodeStrokeColor?: (node: Node) => string;
  getNodeLabelColor?: (node: Node) => string;
  getNodeLabelStrokeColor?: (node: Node) => string;

  getLinkLabel?: (node: Link) => string | undefined;
  getLinkDetails?: (node: Link) => string | undefined;
  getLinkMeta?: (node: Link) => any;

  getLinkColor?: (link: Link) => string;
  getLinkLabelColor?: (link: Link) => string;
  getLinkLabelStrokeColor?: (link: Link) => string;

  onClickNode?: (node: Node, event: MouseEvent) => void;
  onClickLink?: (node: Link, event: MouseEvent) => void;
  onClickCanvas?: () => void;
}

