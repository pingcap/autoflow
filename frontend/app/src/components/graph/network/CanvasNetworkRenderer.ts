import * as d3 from 'd3';
import ForceGraph from 'force-graph';
import type { IdType, NetworkLink, NetworkNode, ReadonlyNetwork } from './Network';
import type { NetworkRendererOptions, NetworkNodeView, NetworkLinkView } from './NetworkRenderer';

export class CanvasNetworkRenderer<Node extends NetworkNode, Link extends NetworkLink> {
  private _el: HTMLElement | undefined;
  private _graph: any; // ForceGraph instance
  private _ro: ResizeObserver | undefined;

  private _onFocusNode: ((id: IdType) => void) | undefined;
  private _onFocusLink: ((id: IdType) => void) | undefined;
  private _onBlurNode: (() => void) | undefined;
  private _onBlurLink: (() => void) | undefined;

  private _onUpdateLink: ((id: IdType) => void) | undefined;
  private _onUpdateNode: ((id: IdType) => void) | undefined;

  private nodes: NetworkNodeView[] = [];
  private links: NetworkLinkView[] = [];

  // Graph state
  private selectedNode: NetworkNodeView | null = null;
  private selectedLink: NetworkLinkView | null = null;
  private highlightedNodes = new Set<IdType>();
  private highlightedLinks = new Set<IdType>();


  private readonly linkDefaultDistance = 30;
  private readonly chargeDefaultStrength = -80;
  private readonly linkHighlightDistance = 120;
  private readonly chargeHighlightStrength = -200;
  private readonly linkDefaultWidth = 1;

  // Clustering
  private clusterMode = 'enabled';
  private clustersCalculated = false;


  // Colors
  private colors = {
    textColor: '#000000',
    nodeColor: '#1f77b4',
    nodeHighlighted: '#18a0b1',
    nodeSelected: '#72fefb',
    linkColor: '#999999',
    linkHighlighted: '#18a0b1',
    linkSelected: '#72fefb',
    clusterColors: [
      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
  };

  scale = 1;

  constructor(
    private network: ReadonlyNetwork<Node, Link>,
    private options: NetworkRendererOptions<Node, Link>,
  ) {
    this.compile(options);
  }

  private compile(options: NetworkRendererOptions<Node, Link>) {
    const nodeMap = new Map<IdType, number>();
    this.nodes = this.network.nodes().map((node, index) => {
      nodeMap.set(node.id, index);
      return {
        id: node.id,
        index,
        radius: 8, // Fixed size like graph.js
        label: options.getNodeLabel?.(node),
        details: options.getNodeDetails?.(node),
        meta: options.getNodeMeta?.(node),
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

  get massive() {
    return this.nodes.length > 50 || this.links.length > 50;
  }

  mount(container: HTMLElement) {
    if (this._el) {
      return;
    }
    this._el = container;

    // Get initial container dimensions
    const { width: initialWidth, height: initialHeight } = container.getBoundingClientRect();

    // Initialize ForceGraph
    this._graph = (ForceGraph as any)()(container)
      .width(initialWidth)
      .height(initialHeight)
      .backgroundColor('transparent')
      .nodeCanvasObject((node: any, ctx: CanvasRenderingContext2D) => {
        this.drawNodeWithLabel(node, ctx);
      })
      .linkWidth(this.linkDefaultWidth)
      .linkColor((link: any) => {
        if (this.selectedLink && this.selectedLink.id === link.id) {
          return this.colors.linkSelected;
        } else if (this.highlightedLinks.has(link.id)) {
          return this.colors.linkHighlighted;
        } else {
          return this.colors.linkColor;
        }
      })
      .linkDirectionalArrowLength(6)
      .linkDirectionalArrowRelPos(1)
      .linkCurvature(0.1)
      .onNodeClick((node: any, event: MouseEvent) => {
        this.onNodeClick(node, event);
      })
      .onLinkClick((link: any, event: MouseEvent) => {
        this.onLinkClick(link, event);
      })
      .onBackgroundClick(() => {
        this.onBackgroundClick();
      })
      .d3Force('x', d3.forceX(0).strength(0.05))
      .d3Force('y', d3.forceY(0).strength(0.05))
      .d3Force("link", d3.forceLink().id((d: any) => d.id).distance(this.linkDefaultDistance))
      .d3Force("charge", d3.forceManyBody().strength(this.chargeDefaultStrength));

    // Set container styling to match legacy renderer
    container.style.overflow = 'hidden';

    // Handle resize
    this._ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      this._graph.width(width).height(height);
    });
    this._ro.observe(container);

    // Set initial data
    this.render();

    // Ensure the canvas fills the container after ForceGraph creates it
    setTimeout(() => {
      const canvas = container.querySelector('canvas');
      if (canvas) {
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.display = 'block';
      }
    }, 0);
  }

  unmount() {
    if (this._onUpdateLink) {
      this.network.off('update:link', this._onUpdateLink);
      this._onUpdateLink = undefined;
    }
    if (this._onUpdateNode) {
      this.network.off('update:node', this._onUpdateNode);
      this._onUpdateNode = undefined;
    }
    if (!this._el) {
      return;
    }
    this._ro?.disconnect();
    this._ro = undefined;
    this._el = undefined;
  }

  private drawNodeWithLabel(node: any, ctx: CanvasRenderingContext2D) {
    const nodeRadius = 8; // Fixed radius like graph.js
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI, false);
    
    const clusterModeOn = this.clusterMode === 'enabled';
    let nodeColor;
    if (clusterModeOn && node.clusterId !== undefined) {
      const clusterId = node.clusterId || 0;
      nodeColor = this.colors.clusterColors[clusterId % this.colors.clusterColors.length];
    } else {
      nodeColor = this.options.getNodeColor?.(this.network.node(node.id) as any) ?? this.colors.nodeColor;
    }
    
    ctx.fillStyle = nodeColor;
    ctx.fill();

    // Add outline for selected or highlighted nodes
    if (this.selectedNode && this.selectedNode.id === node.id) {
      ctx.strokeStyle = this.colors.nodeSelected;
      ctx.lineWidth = 3;
      ctx.stroke();
    } else if (this.highlightedNodes.has(node.id)) {
      ctx.strokeStyle = this.colors.nodeHighlighted;
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    // Draw label
    const label = this.options.getNodeLabel?.(this.network.node(node.id)!) ?? node.name ?? node.id;
    const fontSize = Math.max(8, nodeRadius * 0.3);
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = this.options.getNodeLabelColor?.(this.network.node(node.id) as any) ?? this.colors.textColor;
    ctx.fillText(label, node.x, node.y + nodeRadius + fontSize * 0.7);
  }

  private onNodeClick(node: any, event: MouseEvent) {
    this.selectedNode = node;
    this.selectedLink = null;
    
    this.options.onClickNode?.(this.network.node(node.id)!, event);
    this.highlightConnections(node);
  }

  private onLinkClick(link: any, event: MouseEvent) {
    this.selectedLink = link;
    this.selectedNode = null;
    
    this.options.onClickLink?.(this.network.link(link.id)!, event);
    this.highlightLink(link);
  }

  private onBackgroundClick() {
    this.options.onClickCanvas?.();
    this.clearHighlight();
  }

  private highlightLink(link: any) {
    this.highlightedLinks.clear();
    this.highlightedLinks.add(link.id);
  }

  private highlightConnections(node: any) {
    const connectedNodeIds = new Set<IdType>();
    const connectedLinkIds = new Set<IdType>();
    
    this._graph.graphData().links.forEach((link: any) => {
      if (link.source.id === node.id) {
        connectedNodeIds.add(link.target.id);
        connectedLinkIds.add(link.id);
      } else if (link.target.id === node.id) {
        connectedNodeIds.add(link.source.id);
        connectedLinkIds.add(link.id);
      }
    });
    
    this.highlightedNodes.clear();
    connectedNodeIds.forEach(nodeId => this.highlightedNodes.add(nodeId));
    
    this.highlightedLinks.clear();
    connectedLinkIds.forEach(linkId => this.highlightedLinks.add(linkId));
    
    // Update force simulation for highlighting
    this._graph.d3Force("link").distance((link: any) => {
      if (connectedLinkIds.has(link.id)) {
        return this.linkHighlightDistance;
      }
      return this.linkDefaultDistance;
    });

    this._graph.d3Force("charge").strength((node: any) => {
      if (connectedNodeIds.has(node.id)) {
        return this.chargeHighlightStrength;
      }
      return this.chargeDefaultStrength;
    });

    this._graph.d3ReheatSimulation();
  }

  private clearHighlight() {
    this.selectedNode = null;
    this.selectedLink = null;
    this.highlightedNodes.clear();
    this.highlightedLinks.clear();
    this._graph.d3Force("link").distance(this.linkDefaultDistance);
    this._graph.d3Force("charge").strength(this.chargeDefaultStrength);
  }

  private calculateAndCacheClusters() {
    if (!this.nodes || !this.links) {
      return;
    }
    const clusters = this.findClusters();
    
    this.nodes.forEach(node => {
      (node as any).clusterId = clusters.get(node.id) || 0;
    });
    
    this.clustersCalculated = true;
  }

  private findClusters(): Map<IdType, number> {
    const clusters = new Map<IdType, number>();
    const visited = new Set<IdType>();
    let clusterId = 0;
    
    const adjacencyList = new Map<IdType, IdType[]>();
    this.nodes.forEach(node => {
      adjacencyList.set(node.id, []);
    });
    
    this.links.forEach(link => {
      const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
      const targetId = typeof link.target === 'object' ? link.target.id : link.target;
      
      if (adjacencyList.has(sourceId) && adjacencyList.has(targetId)) {
        adjacencyList.get(sourceId)!.push(targetId);
        adjacencyList.get(targetId)!.push(sourceId);
      }
    });
    
    const dfs = (nodeId: IdType, currentClusterId: number) => {
      if (visited.has(nodeId)) return;
      
      visited.add(nodeId);
      clusters.set(nodeId, currentClusterId);
      
      const neighbors = adjacencyList.get(nodeId) || [];
      neighbors.forEach(neighborId => {
        if (!visited.has(neighborId)) {
          dfs(neighborId, currentClusterId);
        }
      });
    };

    this.nodes.forEach(node => {
      if (!visited.has(node.id)) {
        dfs(node.id, clusterId);
        clusterId++;
      }
    });
    
    return clusters;
  }

  focusNode(id: IdType): void {
    // Find the node and set it as selected
    const node = this.nodes.find(n => n.id === id);
    if (node) {
      this.selectedNode = node;
      this.selectedLink = null;
      this.highlightConnections(node);
      // Trigger visual update
      this._graph.nodeColor(this._graph.nodeColor());
      this._graph.linkColor(this._graph.linkColor());
    }
    this._onFocusNode?.(id);
  }

  blurNode(): void {
    this.clearHighlight();
    // Trigger visual update
    this._graph.nodeColor(this._graph.nodeColor());
    this._graph.linkColor(this._graph.linkColor());
    this._onBlurNode?.();
  }

  focusLink(id: IdType): void {
    // Find the link and set it as selected
    const link = this.links.find(l => l.id === id);
    if (link) {
      this.selectedLink = link;
      this.selectedNode = null;
      this.highlightLink(link);
      // Trigger visual update
      this._graph.nodeColor(this._graph.nodeColor());
      this._graph.linkColor(this._graph.linkColor());
    }
    this._onFocusLink?.(id);
  }

  blurLink(): void {
    this.clearHighlight();
    // Trigger visual update
    this._graph.nodeColor(this._graph.nodeColor());
    this._graph.linkColor(this._graph.linkColor());
    this._onBlurLink?.();
  }

  render() {
    // Calculate clusters if needed
    if (!this.clustersCalculated) {
      this.calculateAndCacheClusters();
    }

    // Set graph data
    const graphData = {
      nodes: this.nodes,
      links: this.links
    };
    
    this._graph.graphData(graphData);

    // Set up event handlers
    this._onFocusNode = () => {
      // Focus logic would go here
    };

    this._onFocusLink = () => {
      // Focus logic would go here  
    };

    this._onBlurNode = () => {
      this.clearHighlight();
    };

    this._onBlurLink = () => {
      this.clearHighlight();
    };

    this._onUpdateNode = () => {
      // Update node logic
    };

    this._onUpdateLink = () => {
      // Update link logic
    };

    this.network.on('update:node', this._onUpdateNode);
    this.network.on('update:link', this._onUpdateLink);

    // Zoom to fit after initial render
    setTimeout(() => {
      this._graph.zoomToFit(400, 50);
    }, 1000);
  }
}