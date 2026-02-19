/**
 * D3 Helper Utilities - Force simulation and viewport management
 *
 * US-1402.1: Interactive Mesh D3 UI
 * Epic AC: Performance optimization for 1000+ nodes
 *
 * JPL Power of Ten Compliance:
 * - Rule #9: Complete type hints
 * - Rule #4: All functions < 60 lines
 * - Rule #2: Fixed upper bounds
 */

import * as d3 from 'd3';

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

/** Node with D3 simulation properties */
export interface SimulationNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  type: string;
  [key: string]: any;
}

/** Edge with D3 simulation properties */
export interface SimulationEdge
  extends d3.SimulationLinkDatum<SimulationNode> {
  source: string | SimulationNode;
  target: string | SimulationNode;
  type?: string;
  weight?: number;
}

/** Force simulation configuration */
export interface ForceConfig {
  linkDistance?: number;
  linkStrength?: number;
  chargeStrength?: number;
  collisionRadius?: number;
  centerStrength?: number;
  alphaDecay?: number;
}

/** Viewport bounds */
export interface ViewportBounds {
  x: number;
  y: number;
  width: number;
  height: number;
  scale: number;
}

/** Zoom configuration */
export interface ZoomConfig {
  minScale?: number;
  maxScale?: number;
  duration?: number;
}

// =============================================================================
// CONSTANTS (Rule #2: Fixed upper bounds)
// =============================================================================

const DEFAULT_LINK_DISTANCE = 120;
const DEFAULT_LINK_STRENGTH = 0.7;
const DEFAULT_CHARGE_STRENGTH = -250;
const DEFAULT_COLLISION_RADIUS = 40;
const DEFAULT_CENTER_STRENGTH = 0.1;
const DEFAULT_ALPHA_DECAY = 0.02;

const MIN_ZOOM_SCALE = 0.1;
const MAX_ZOOM_SCALE = 10;
const DEFAULT_ZOOM_DURATION = 500;

const CULLING_MARGIN = 100; // Extra margin for viewport culling

/** Maximum nodes to process for culling (JPL Rule #2) */
const MAX_CULL_NODES = 10000;

// =============================================================================
// FORCE SIMULATION (Rule #4: < 60 lines each)
// =============================================================================

/**
 * Create D3 force simulation with configuration.
 *
 * US-1402.1: Interactive Mesh D3 UI
 * Epic AC: Smooth 60fps interaction.
 *
 * @param nodes - Graph nodes
 * @param edges - Graph edges
 * @param width - Viewport width
 * @param height - Viewport height
 * @param config - Optional force configuration
 * @returns Configured D3 simulation
 */
export function createForceSimulation(
  nodes: SimulationNode[],
  edges: SimulationEdge[],
  width: number,
  height: number,
  config: ForceConfig = {}
): d3.Simulation<SimulationNode, SimulationEdge> {
  const {
    linkDistance = DEFAULT_LINK_DISTANCE,
    linkStrength = DEFAULT_LINK_STRENGTH,
    chargeStrength = DEFAULT_CHARGE_STRENGTH,
    collisionRadius = DEFAULT_COLLISION_RADIUS,
    centerStrength = DEFAULT_CENTER_STRENGTH,
    alphaDecay = DEFAULT_ALPHA_DECAY,
  } = config;

  return d3
    .forceSimulation<SimulationNode>(nodes)
    .force(
      'link',
      d3
        .forceLink<SimulationNode, SimulationEdge>(edges)
        .id((d) => d.id)
        .distance(linkDistance)
        .strength(linkStrength)
    )
    .force('charge', d3.forceManyBody().strength(chargeStrength))
    .force(
      'center',
      d3.forceCenter(width / 2, height / 2).strength(centerStrength)
    )
    .force('collision', d3.forceCollide().radius(collisionRadius))
    .alphaDecay(alphaDecay);
}

/**
 * Create zoom behavior for SVG.
 *
 * US-1402.1: Pan & Zoom requirement.
 * JPL Rule #4: Function < 60 lines.
 *
 * @param svg - SVG selection
 * @param container - Container group selection
 * @param config - Zoom configuration
 * @returns Zoom behavior
 */
export function createZoomBehavior<
  SVGElement extends Element,
  Datum = unknown
>(
  svg: d3.Selection<SVGElement, Datum, null, undefined>,
  container: d3.Selection<SVGGElement, unknown, null, undefined>,
  config: ZoomConfig = {}
): d3.ZoomBehavior<SVGElement, Datum> {
  const {
    minScale = MIN_ZOOM_SCALE,
    maxScale = MAX_ZOOM_SCALE,
  } = config;

  const zoom = d3
    .zoom<SVGElement, Datum>()
    .scaleExtent([minScale, maxScale])
    .on('zoom', (event: d3.D3ZoomEvent<SVGElement, Datum>) => {
      requestAnimationFrame(() => {
        container.attr('transform', event.transform.toString());
      });
    });

  svg.call(zoom);
  return zoom;
}

/**
 * Center viewport on a specific node.
 *
 * US-1402.1: Double-click to center.
 * JPL Rule #4: Function < 60 lines.
 *
 * @param svg - SVG selection
 * @param zoom - Zoom behavior
 * @param node - Node to center on
 * @param width - Viewport width
 * @param height - Viewport height
 * @param scale - Target zoom scale
 */
export function centerOnNode<SVGElement extends Element, Datum>(
  svg: d3.Selection<SVGElement, Datum, null, undefined>,
  zoom: d3.ZoomBehavior<SVGElement, Datum>,
  node: SimulationNode,
  width: number,
  height: number,
  scale: number = 1.5
): void {
  const x = width / 2 - (node.x || 0) * scale;
  const y = height / 2 - (node.y || 0) * scale;

  svg
    .transition()
    .duration(DEFAULT_ZOOM_DURATION)
    .call(
      zoom.transform as any,
      d3.zoomIdentity.translate(x, y).scale(scale)
    );
}

/**
 * Fit entire graph to viewport.
 *
 * US-1402.1: Reset view functionality.
 * JPL Rule #4: Function < 60 lines.
 *
 * @param svg - SVG selection
 * @param zoom - Zoom behavior
 * @param nodes - All nodes
 * @param width - Viewport width
 * @param height - Viewport height
 * @param padding - Padding around graph (default 50px)
 */
export function fitGraphToViewport<SVGElement extends Element, Datum>(
  svg: d3.Selection<SVGElement, Datum, null, undefined>,
  zoom: d3.ZoomBehavior<SVGElement, Datum>,
  nodes: SimulationNode[],
  width: number,
  height: number,
  padding: number = 50
): void {
  if (nodes.length === 0) return;

  // Calculate bounding box
  const xs = nodes.map((n) => n.x || 0);
  const ys = nodes.map((n) => n.y || 0);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const graphWidth = maxX - minX;
  const graphHeight = maxY - minY;

  // Calculate scale to fit
  const scale = Math.min(
    (width - padding * 2) / graphWidth,
    (height - padding * 2) / graphHeight,
    MAX_ZOOM_SCALE
  );

  // Calculate translation
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const x = width / 2 - centerX * scale;
  const y = height / 2 - centerY * scale;

  svg
    .transition()
    .duration(DEFAULT_ZOOM_DURATION)
    .call(
      zoom.transform as any,
      d3.zoomIdentity.translate(x, y).scale(scale)
    );
}

// =============================================================================
// VIEWPORT CULLING (Rule #4: < 60 lines each)
// =============================================================================

/**
 * Filter nodes to only those visible in viewport.
 *
 * US-1402.1: Viewport culling optimization.
 * Epic AC: Maintain 60fps with 1000+ nodes.
 * JPL Rule #2: Bounded operation (max MAX_CULL_NODES).
 *
 * @param nodes - All nodes (will process up to MAX_CULL_NODES)
 * @param viewport - Current viewport bounds
 * @returns Visible nodes only
 */
export function cullNodesOutsideViewport(
  nodes: SimulationNode[],
  viewport: ViewportBounds
): SimulationNode[] {
  const { x, y, width, height, scale } = viewport;

  // Calculate visible area in graph coordinates
  const minX = -x / scale - CULLING_MARGIN;
  const maxX = (-x + width) / scale + CULLING_MARGIN;
  const minY = -y / scale - CULLING_MARGIN;
  const maxY = (-y + height) / scale + CULLING_MARGIN;

  // JPL Rule #2: Enforce fixed upper bound on nodes to process
  const boundedNodes = nodes.slice(0, MAX_CULL_NODES);

  // Filter with bounded array
  return boundedNodes.filter((node) => {
    const nx = node.x || 0;
    const ny = node.y || 0;
    return nx >= minX && nx <= maxX && ny >= minY && ny <= maxY;
  });
}

/**
 * Get viewport bounds from D3 zoom transform.
 *
 * US-1402.1: Viewport tracking.
 * JPL Rule #4: Function < 60 lines.
 *
 * @param transform - D3 zoom transform
 * @param width - Viewport width
 * @param height - Viewport height
 * @returns Viewport bounds
 */
export function getViewportBounds(
  transform: d3.ZoomTransform,
  width: number,
  height: number
): ViewportBounds {
  return {
    x: transform.x,
    y: transform.y,
    width,
    height,
    scale: transform.k,
  };
}

// =============================================================================
// UTILITY FUNCTIONS (Rule #4: < 60 lines each)
// =============================================================================

/**
 * Calculate node size based on properties.
 *
 * US-1402.1: Node size proportional to importance.
 * JPL Rule #4: Function < 60 lines.
 *
 * @param citationCount - Number of citations
 * @param minSize - Minimum node size (default 10)
 * @param maxSize - Maximum node size (default 50)
 * @returns Calculated node size
 */
export function calculateNodeSize(
  citationCount: number,
  minSize: number = 10,
  maxSize: number = 50
): number {
  // Logarithmic scaling for better visual distribution
  const scale = Math.log(citationCount + 1) * 5;
  return Math.max(minSize, Math.min(maxSize, minSize + scale));
}

/**
 * Truncate label to maximum length.
 *
 * US-1402.1: No overlapping labels.
 * JPL Rule #4: Function < 60 lines.
 *
 * @param label - Original label
 * @param maxLength - Maximum length (default 15)
 * @returns Truncated label with ellipsis if needed
 */
export function truncateLabel(
  label: string,
  maxLength: number = 15
): string {
  if (label.length <= maxLength) {
    return label;
  }
  return label.slice(0, maxLength) + '...';
}

/**
 * Throttle function calls.
 *
 * US-1402.1: Performance optimization.
 * JPL Rule #4: Function < 60 lines.
 *
 * @param func - Function to throttle
 * @param delay - Delay in milliseconds
 * @returns Throttled function
 */
export function throttle<T extends (...args: any[]) => void>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let lastCall = 0;
  return (...args: Parameters<T>) => {
    const now = Date.now();
    if (now - lastCall >= delay) {
      lastCall = now;
      func(...args);
    }
  };
}

/**
 * Debounce function calls.
 *
 * US-1402.1: Search debouncing.
 * JPL Rule #4: Function < 60 lines.
 *
 * @param func - Function to debounce
 * @param delay - Delay in milliseconds
 * @returns Debounced function
 */
export function debounce<T extends (...args: any[]) => void>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout | null = null;

  return (...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      func(...args);
      timeoutId = null;
    }, delay);
  };
}
