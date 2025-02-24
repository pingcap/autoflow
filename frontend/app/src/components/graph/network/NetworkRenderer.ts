import type { SimulationLinkDatum, SimulationNodeDatum } from 'd3';
import * as d3 from 'd3';
import type { IdType, NetworkLink, NetworkNode, ReadonlyNetwork } from './Network';

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

export interface NetworkNodeView extends SimulationNodeDatum {
  id: IdType;
  index: number;
  radius: number;
  label?: string;
  details?: string;
  meta?: any;
}

export interface NetworkLinkView extends SimulationLinkDatum<NetworkNodeView> {
  id: IdType;
  index: number;
  source: NetworkNodeView;
  target: NetworkNodeView;
  label?: string;
  details?: string;
  meta?: any;
}

export class NetworkRenderer<Node extends NetworkNode, Link extends NetworkLink> {
  private _el: HTMLElement | undefined;
  private _graph: d3.Selection<SVGSVGElement, undefined, null, undefined>;
  private _grid: d3.Selection<SVGGElement, undefined, null, undefined>;
  private _ro: ResizeObserver | undefined;
  private _gridZoomed: ((event: d3.D3ZoomEvent<any, any>) => void) | undefined;
  private _destroyGrid: (() => void) | undefined;

  private _onFocusNode: ((id: IdType) => void) | undefined;
  private _onFocusLink: ((id: IdType) => void) | undefined;
  private _onBlurNode: (() => void) | undefined;
  private _onBlurLink: (() => void) | undefined;

  private _onUpdateLink: ((id: IdType) => void) | undefined;
  private _onUpdateNode: ((id: IdType) => void) | undefined;

  private nodes: NetworkNodeView[] = [];
  private links: NetworkLinkView[] = [];

  private canvas: HTMLCanvasElement | undefined;
  scale = 1;

  constructor (
    private network: ReadonlyNetwork<Node, Link>,
    private options: NetworkRendererOptions<Node, Link>,
  ) {
    ;(window as any).network = network;

    this.compile(options);

    this._graph = d3.create('svg')
      .attr('style', 'max-width: 100%; height: auto; height: intrinsic;');

    this._grid = this._graph
      .append('g');
  }

  private compile (options: NetworkRendererOptions<Node, Link>) {
    const nodeMap = new Map<IdType, number>();
    this.nodes = this.network.nodes().map((node, index) => {
      nodeMap.set(node.id, index);
      return {
        id: node.id,
        index,
        radius: options.getNodeRadius?.(node) ?? 5,
        label: options.getNodeLabel?.(node),
        details: options.getNodeDetails?.(node),
        meta: options.getNodeMeta?.(node),
        // x, y
        ...options.getNodeInitialAttrs?.(node, index),
      };
    });
    this.links = this.network.links().map((link, index) => ({
      id: link.id,
      index,
      source: this.nodes[nodeMap.get(link.source)!],
      target: this.nodes[nodeMap.get(link.target)!],
      label: options.getLinkLabel?.(link),
      details: options.getLinkDetails?.(link),
      meta: options.getLinkMeta?.(link),
    }));
  }

  get massive () {
    return this.nodes.length > 50 || this.links.length > 50;
  }

  mount (container: HTMLElement) {
    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
    }

    if (this._el) {
      return;
    }
    this._el = container;
    const ro = this._ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      this._graph
        .attr('width', width + 'px')
        .attr('height', height + 'px')
        .attr('viewBox', `${-width / 2} ${-height / 2} ${width} ${height}`);

      this.renderGrid(width, height);
    });
    ro.observe(container);
    container.style.overflow = 'hidden';
    container.append(this._graph.node()!);
    const { width, height } = container.getBoundingClientRect();
    this.render();
    this.renderGrid(width, height);
  }

  unmount () {
    if (this._onUpdateLink) {
      this.network.on('update:link', this._onUpdateLink);
      this._onUpdateLink = undefined;
    }
    if (this._onUpdateNode) {
      this.network.on('update:node', this._onUpdateNode);
      this._onUpdateNode = undefined;
    }
    if (!this._el) {
      return;
    }
    this._el.removeChild(this._graph.node()!);
    this._ro?.disconnect();
    this._ro = undefined;
  }

  private renderGrid (width: number, height: number) {
    this._destroyGrid?.();

    const svg = this._grid;
    const k = height / width;

    const x = d3.scaleLinear()
      .domain([-4.5, 4.5])
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([-4.5 * k, 4.5 * k])
      .range([height, 0]);

    const xAxis = (g: d3.Selection<SVGGElement, any, any, any>, x: d3.ScaleLinear<any, any>) => g
      .attr('transform', `translate(${-width / 2},${height / 2})`)
      .call(d3.axisTop(x).ticks(12))
      .call(g => g.select('.domain').attr('display', 'none'));

    const yAxis = (g: d3.Selection<SVGGElement, any, any, any>, y: d3.ScaleLinear<any, any>) => g
      .attr('transform', `translate(-${width / 2},-${height / 2})`)
      .call(d3.axisRight(y).ticks(12 * k))
      .call(g => g.select('.domain').attr('display', 'none'));

    const grid = (g: d3.Selection<SVGGElement, any, any, any>, x: d3.ScaleLinear<any, any>, y: d3.ScaleLinear<any, any>) => g
      .attr('transform', `translate(-${width / 2},-${height / 2})`)
      .attr('stroke', 'currentColor')
      .attr('stroke-opacity', 0.1)
      .call(g => g
        .selectAll('.x')
        .data(x.ticks(12))
        .join(
          enter => enter.append('line').attr('class', 'x').attr('y2', height),
          update => update,
          exit => exit.remove(),
        )
        .attr('x1', d => 0.5 + x(d))
        .attr('x2', d => 0.5 + x(d)))
      .call(g => g
        .selectAll('.y')
        .data(y.ticks(12 * k))
        .join(
          enter => enter.append('line').attr('class', 'y').attr('x2', width),
          update => update,
          exit => exit.remove(),
        )
        .attr('y1', d => 0.5 + y(d))
        .attr('y2', d => 0.5 + y(d)));

    const gGrid = svg.append('g');

    const gx = svg.append('g');

    const gy = svg.append('g');

    gGrid.call(grid, d3.zoomIdentity.rescaleX(x), d3.zoomIdentity.rescaleY(y));

    this._gridZoomed = ({ transform }) => {
      const zx = transform.rescaleX(x).interpolate(d3.interpolateRound);
      const zy = transform.rescaleY(y).interpolate(d3.interpolateRound);

      gx.call(xAxis, zx);
      gy.call(yAxis, zy);
      gGrid.call(grid, zx, zy);
    };

    this._destroyGrid = () => {
      gx.remove();
      gy.remove();
      gGrid.remove();
    };
  }

  focusNode (id: IdType): void {
    this._onFocusNode?.(id);
  }

  blurNode (): void {
    this._onBlurNode?.();
  }

  focusLink (id: IdType): void {
    this._onFocusLink?.(id);

  }

  blurLink (): void {
    this._onBlurLink?.();
  }

  render () {
    // @ts-ignore
    let currentLinkId: IdType | undefined = undefined;
    // @ts-ignore
    let currentNodeId: IdType | undefined = undefined;

    const svg = this._graph
      .on('click', () => {
        this.options.onClickCanvas?.();
      });

    const link = svg.append('g')
      .attr('opacity', 1)
      .attr('stroke-linejoin', 'round')
      .selectAll()
      .data(this.links)
      .join('g')
      .style('cursor', 'pointer')
      .on('click', (event, link) => {
        event.stopPropagation();
        this.options.onClickLink?.(this.network.link(link.id)!, event);
      });

    const linkPath = link
      .append('path')
      .attr('fill', d => this.options.getLinkColor?.(this.network.link(d.id) as any) ?? 'hsl(var(--primary) / 50%)');

    let linkText = (this.options.showLinkLabel !== false && !this.massive)
      ? link
        .append('text')
        .attr('font-size', 12)
        .attr('fill', d => this.options.getLinkLabelColor?.(this.network.link(d.id) as any) ?? 'hsl(var(--primary))')
        .attr('stroke', d => this.options.getLinkLabelStrokeColor?.(this.network.link(d.id) as any) ?? 'hsl(var(--primary-foreground))')
        .attr('stroke-width', 2)
        .attr('paint-order', 'stroke')
        .attr('text-anchor', 'middle')
        .text(d => this.options.getLinkLabel?.(this.network.link(d.id)!) ?? '')
      : undefined;

    const node = svg.append('g')
      .selectAll('g')
      .data(this.nodes)
      .enter()
      .append('g')
      .attr('title', d => d.details ?? null)
      .style('cursor', 'pointer')
      .attr('opacity', 1)
      .on('click', (event, node) => {
        event.stopPropagation();
        this.options.onClickNode?.(this.network.node(node.id)!, event);
      });

    const circle = node
      .append('circle')
      .attr('stroke', d => this.options.getNodeStrokeColor?.(this.network.node(d.id) as any) ?? 'hsl(var(--primary))')
      .attr('fill', d => this.options.getNodeColor?.(this.network.node(d.id) as any) ?? 'hsl(var(--primary))')
      .attr('r', node => node.radius / this.scale);

    const text = node
      .append('text')
      .text((d) => this.createText(d.id))
      .attr('fill', d => this.options.getNodeLabelColor?.(this.network.node(d.id) as any) ?? 'hsl(var(--primary))')
      .attr('stroke', d => this.options.getNodeLabelStrokeColor?.(this.network.node(d.id) as any) ?? 'hsl(var(--primary-foreground))')
      .attr('stroke-width', 2)
      .attr('font-size', 12)
      .attr('paint-order', 'stroke')
      .attr('alignment-baseline', 'center')
      .attr('opacity', 1);

    const ticked = () => {
      linkPath.attr('d', (node) => this.createLinkPath(node.source, node.target, 1));

      linkText?.call((selection) => {
        this.assignLinkText(selection, this.scale);
      });

      circle
        .attr('cx', d => d.x!)
        .attr('cy', d => d.y!)
        .call(drag(simulation));

      text
        .attr('x', node => this.createTextX(node, this.scale))
        .attr('y', node => this.createTextY(node, this.scale));

    };

    const zoomed = (event: d3.D3ZoomEvent<any, undefined>) => {
      const { transform } = event;
      if (isFinite(transform.k)) {
        this.scale = transform.k;
      }

      circle.attr('transform', transform as any);
      // .attr('r', node => node.radius / transform.k)
      // .attr('stroke-width', 1 / transform.k);

      link.attr('transform', transform as any);

      // linkPath.attr('transform', transform as any);
      // .attr('stroke-width', 3 / transform.k)
      // .attr('d', (node) => this.createLinkPath(node.source, node.target, transform.k));

      linkText
        ?.attr('font-size', 12 / transform.k)
        .attr('stroke-width', 2 / transform.k)
        .call((selection) => {
          this.assignLinkText(selection, transform.k);
        });

      link.append('title')
        .text(d => this.options.getLinkLabel?.(this.network.link(d.id)!) ?? '');

      text.attr('transform', transform as any)
        .attr('font-size', 12 / transform.k)
        .attr('stroke-width', 2 / transform.k)
        .attr('x', node => this.createTextX(node, transform.k))
        .attr('y', node => this.createTextY(node, transform.k));

      this._gridZoomed?.(event);
    };

    const forceNode = d3.forceCollide<NetworkNodeView>(node => node.radius * 1.5);
    const forceManyBody = d3.forceManyBody<NetworkNodeView>();
    const forceLink = d3.forceLink<NetworkNodeView, NetworkLinkView>(this.links).strength(0.01).id(node => node.id);
    const forceCenter = d3.forceCenter();

    const simulation = d3.forceSimulation(this.nodes)
      .force('link', forceLink)
      .force('charge', forceNode)
      .force('manyBody', forceManyBody)
      .force('center', forceCenter)
      .on('tick', ticked);

    const drag = (simulation: d3.Simulation<NetworkNodeView, NetworkLinkView>) => {
      const dragstarted = (event: d3.D3DragEvent<any, NetworkNodeView, { x: number, y: number, fx: number, fy: number }>) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      };

      const dragged = (event: d3.D3DragEvent<any, NetworkNodeView, { x: number, y: number, fx: number, fy: number }>) => {
        event.subject.fx += event.dx / this.scale;
        event.subject.fy += event.dy / this.scale;
      };

      const dragended = (event: d3.D3DragEvent<any, NetworkNodeView, { x: number, y: number, fx: number | null, fy: number | null }>) => {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      };

      return d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
    };

    const zoom = d3.zoom<any, undefined>()
      .scaleExtent([0.1, 32])
      .on('zoom', zoomed);

    svg
      .call(zoom)
      .call(zoom.transform, d3.zoomIdentity);

    const onFocusLink = (id: IdType) => {
      currentLinkId = id;
      const currentLink = this.network.link(id);
      link.transition().attr('opacity', link => link.id === id ? 1 : 0.2);
      node.transition().attr('opacity', node => {
        return (currentLink?.target === node.id || currentLink?.source === node.id) ? 1 : 0.2;
      });
      if (this.massive) {
        text.transition()
          .attr('opacity', (node) => {
            return (currentLink?.target === node.id || currentLink?.source === node.id) ? 1 : 0;
          });
      }
    };

    const onFocusNode = (id: IdType) => {
      currentNodeId = id;
      const currentNode = this.network.node(id);

      node
        .transition()
        .attr('opacity', node => {
          if (node.id === currentNode?.id) {
            return 1;
          }
          if (this.network.nodeNeighborhoods(id)?.has(node.id)) {
            return 0.7;
          }
          return 0.2;
        });

      if (this.massive) {
        text.transition()
          .attr('opacity', node => {
            if (node.id === currentNode?.id) {
              return 1;
            }
            if (this.network.nodeNeighborhoods(id)?.has(node.id)) {
              return 0.7;
            }
            return 0;
          });
      }

      link
        .transition()
        .attr('opacity', link => {
          return (link.source.id === currentNode?.id || link.target.id === currentNode?.id) ? 1 : 0.2;
        });
    };

    const onBlur = () => {
      currentLinkId = undefined;
      currentNodeId = undefined;
      node.transition().attr('opacity', 1);
      link.transition().attr('opacity', 1);
      if (this.massive) {
        text.transition().attr('opacity', 0);
      }
    };

    this._onFocusNode = onFocusNode;
    this._onFocusLink = onFocusLink;
    this._onBlurNode = onBlur;
    this._onBlurLink = onBlur;

    this._onUpdateLink = () => {
      // noop currently
    };

    this._onUpdateNode = id => {
      text.filter(node => node.id === id).text(this.createText(id));
    };

    this.network.on('update:link', this._onUpdateLink);
    this.network.on('update:node', this._onUpdateNode);
  }

  private createText (id: IdType) {
    const node = this.network.node(id) as any;
    return (node?.name ?? node?.description) + (this.options.showId ? ` #${id}` : '');
  }

  private assignLinkText (text: d3.Selection<SVGTextElement, NetworkLinkView, SVGGElement, undefined>, scale: number) {
    const options = this.options;
    const network = this.network;
    const that = this;
    text.each(function (link) {
      let source = link.source;
      let target = link.target;

      if (source.x! > target.x!) {
        let t = target;
        target = source;
        source = t;
      }

      const theta = Math.atan2(target.x! - source.x!, target.y! - source.y!);
      const x1 = source.x! + Math.sin(theta) * (source.radius + 3) / scale;
      const y1 = source.y! + Math.cos(theta) * (source.radius + 3) / scale;
      const x2 = target.x! - Math.sin(theta) * (target.radius + 3) / scale;
      const y2 = target.y! - Math.cos(theta) * (target.radius + 3) / scale;

      const off = 2 + 6 / scale;

      const dx = x2 - x1;
      const dy = y2 - y1;

      const d = Math.sqrt(dx * dx + dy * dy);

      const nx = dx / d;
      const ny = dy / d;

      const x = x1 + dx / 2 + ny * off;
      const y = y1 + dy / 2 - nx * off;

      const text = options.getLinkLabel?.(network.link(link.id)!) ?? '';

      d3.select(this)
        .attr('x', x)
        .attr('y', y)
        .attr('width', d)
        .attr('transform', `rotate(${-theta / Math.PI * 180 - 270})`)
        .attr('transform-origin', `${x} ${y}`)
        .text(that.clipText(text, d, 12 / scale, 2 / scale));
    });
  }

  private createLinkPath (source: NetworkNodeView, target: NetworkNodeView, scale: number): string {
    const sourceR = source.radius;
    const targetR = target.radius;

    const theta = Math.atan2(target.x! - source.x!, target.y! - source.y!);
    const x1 = source.x! + Math.sin(theta) * (sourceR + 3) / scale;
    const y1 = source.y! + Math.cos(theta) * (sourceR + 3) / scale;
    const x2 = target.x! - Math.sin(theta) * (targetR + 3) / scale;
    const y2 = target.y! - Math.cos(theta) * (targetR + 3) / scale;
    return arrow(x1, y1, x2, y2, 3 / scale);
  }

  private createTextX (node: NetworkNodeView, _scale: number): number {
    return node.x! + (node.radius + 1);
  }

  private createTextY (node: NetworkNodeView, _scale: number): number {
    return node.y! + node.radius / 2;
  }

  private clipText (text: string, width: number, fontSize: number, lineWidth: number): string {
    const ctx = this.canvas!.getContext('2d')!;
    ctx.font = `${fontSize}px sans-serif`;
    ctx.lineWidth = lineWidth;
    const rect = ctx.measureText(text);

    if (compare(rect, width)) {
      return text;
    }

    let min = 0;
    let max = text.length;

    while (max - min > 1) {
      const t = Math.floor((min + max) / 2);
      const rect = ctx.measureText(text.slice(0, t) + '...');
      if (compare(rect, width)) {
        min = t;
      } else {
        max = t;
      }
    }

    if (min < 3) {
      return '';
    }
    return text.slice(0, min) + '...';
  }
}

function compare (metrics: TextMetrics, width: number) {
  return metrics.actualBoundingBoxRight - metrics.actualBoundingBoxLeft < width;
}

function arrow (x1: number, y1: number, x2: number, y2: number, width: number) {
  const dx = x2 - x1;
  const dy = y2 - y1;

  const d = Math.sqrt(dx * dx + dy * dy);

  const nx = dx / d;
  const ny = dy / d;

  const s1x = x1 + ny * width / 2;
  const s1y = y1 - nx * width / 2;

  const s2x = x1 - ny * width / 2;
  const s2y = y1 + nx * width / 2;

  const tbx = x2 - nx * width * 3.5;
  const tby = y2 - ny * width * 3.5;

  const tb1x = tbx + ny * width / 2;
  const tb1y = tby - nx * width / 2;

  const t1x = tbx + ny * width * 2;
  const t1y = tby - nx * width * 2;

  const t2x = tbx - ny * width * 2;
  const t2y = tby + nx * width * 2;

  const tb2x = tbx - ny * width / 2;
  const tb2y = tby + nx * width / 2;

  return [
    'M', x1, y1,
    'L', s1x, s1y,
    'L', tb1x, tb1y,
    'L', t1x, t1y,
    'L', x2, y2,
    'L', t2x, t2y,
    'L', tb2x, tb2y,
    'L', s2x, s2y,
    'z',
  ].join(' ');

}
