"use client";

/**
 * KnowledgeMesh - Interactive Knowledge Graph Visualization
 *
 * US-1402.1: Renders the Knowledge Graph as an interactive force-directed diagram.
 *
 * Features:
 * - D3.js force-directed layout
 * - Nodes colored by entity type
 * - Zoom/Pan support at 60fps
 * - Click node to highlight 1st-hop neighbors
 *
 * JPL Power of Ten Compliance:
 * - Rule #4: All functions < 60 lines
 * - Rule #9: Complete type hints
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

/** Entity types for node coloring */
export type EntityType =
  | 'person'
  | 'organization'
  | 'location'
  | 'concept'
  | 'document'
  | 'entity'
  | 'event'
  | 'date'
  | 'unknown';

/** Node in the knowledge graph */
export interface MeshNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  type: EntityType;
  metadata?: Record<string, unknown>;
}

/** Edge in the knowledge graph */
export interface MeshEdge extends d3.SimulationLinkDatum<MeshNode> {
  source: string | MeshNode;
  target: string | MeshNode;
  label?: string;
  weight?: number;
}

/** Graph data structure */
export interface MeshData {
  nodes: MeshNode[];
  edges: MeshEdge[];
}

/** Props for KnowledgeMesh component */
export interface KnowledgeMeshProps {
  data: MeshData;
  width?: number;
  height?: number;
  onNodeClick?: (node: MeshNode, neighbors: MeshNode[]) => void;
  onNodeHover?: (node: MeshNode | null) => void;
}

// =============================================================================
// CONSTANTS (Rule #2: Fixed upper bounds)
// =============================================================================

/** Color palette by entity type */
const ENTITY_COLORS: Record<EntityType, string> = {
  person: '#e94560',
  organization: '#4fc3f7',
  location: '#66bb6a',
  concept: '#ab47bc',
  document: '#5c6bc0',
  entity: '#ffc107',
  event: '#ff7043',
  date: '#26a69a',
  unknown: '#78909c',
};

const DIM_OPACITY = 0.15;
const FULL_OPACITY = 1.0;
const LINK_HIGHLIGHT_OPACITY = 0.8;
const LINK_DIM_OPACITY = 0.05;
const NODE_RADIUS = 14;
const NODE_HOVER_RADIUS = 18;
const LABEL_MAX_LENGTH = 15;

// =============================================================================
// HELPER FUNCTIONS (Rule #4: < 60 lines each)
// =============================================================================

/** Get color for entity type */
export function getNodeColor(type: EntityType): string {
  return ENTITY_COLORS[type] || ENTITY_COLORS.unknown;
}

/** Build neighbor ID set for a node */
export function buildNeighborIds(nodeId: string, edges: MeshEdge[]): Set<string> {
  const neighbors = new Set<string>();
  for (const edge of edges) {
    const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
    const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;
    if (sourceId === nodeId) neighbors.add(targetId);
    else if (targetId === nodeId) neighbors.add(sourceId);
  }
  return neighbors;
}

/** Truncate label if too long */
export function truncateLabel(label: string): string {
  return label.length > LABEL_MAX_LENGTH
    ? label.slice(0, LABEL_MAX_LENGTH) + '...'
    : label;
}

/** Setup zoom behavior on SVG */
function setupZoom(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  container: d3.Selection<SVGGElement, unknown, null, undefined>
): d3.ZoomBehavior<SVGSVGElement, unknown> {
  const zoom = d3.zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.1, 4])
    .on("zoom", (event) => {
      requestAnimationFrame(() => {
        container.attr("transform", event.transform);
      });
    });
  svg.call(zoom as any);
  return zoom;
}

/** Create force simulation */
function createSimulation(
  nodes: MeshNode[],
  edges: MeshEdge[],
  width: number,
  height: number
): d3.Simulation<MeshNode, MeshEdge> {
  return d3.forceSimulation<MeshNode>(nodes)
    .force("link", d3.forceLink<MeshNode, MeshEdge>(edges)
      .id(d => d.id)
      .distance(120))
    .force("charge", d3.forceManyBody().strength(-250))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(40))
    .alphaDecay(0.02);
}

/** Render edges to container */
function renderEdges(
  container: d3.Selection<SVGGElement, unknown, null, undefined>,
  edges: MeshEdge[]
): d3.Selection<SVGLineElement, MeshEdge, SVGGElement, unknown> {
  const edgeGroup = container.append("g").attr("class", "edges");
  return edgeGroup
    .selectAll("line")
    .data(edges)
    .join("line")
    .attr("stroke", "#4b5563")
    .attr("stroke-opacity", 0.4)
    .attr("stroke-width", d => Math.max(1, (d.weight || 1) * 0.5));
}

/** Render nodes to container */
function renderNodes(
  container: d3.Selection<SVGGElement, unknown, null, undefined>,
  nodes: MeshNode[],
  simulation: d3.Simulation<MeshNode, MeshEdge>
): d3.Selection<SVGGElement, MeshNode, SVGGElement, unknown> {
  const nodeGroup = container.append("g").attr("class", "nodes");
  const nodeElements = nodeGroup
    .selectAll<SVGGElement, MeshNode>("g")
    .data(nodes)
    .join("g")
    .attr("class", "node-group")
    .style("cursor", "pointer")
    .call(createDragBehavior(simulation) as any);

  // Add circles
  nodeElements.append("circle")
    .attr("r", NODE_RADIUS)
    .attr("fill", d => getNodeColor(d.type))
    .attr("stroke", "#1f2937")
    .attr("stroke-width", 2);

  // Add labels
  nodeElements.append("text")
    .text(d => truncateLabel(d.label))
    .attr("x", 18)
    .attr("y", 4)
    .attr("fill", "#e5e7eb")
    .style("font-size", "11px")
    .style("font-weight", "500")
    .style("pointer-events", "none");

  return nodeElements;
}

/** Create drag behavior for nodes */
function createDragBehavior(
  simulation: d3.Simulation<MeshNode, MeshEdge>
): d3.DragBehavior<SVGGElement, MeshNode, MeshNode> {
  return d3.drag<SVGGElement, MeshNode>()
    .on("start", (event, d) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    })
    .on("drag", (event, d) => {
      d.fx = event.x;
      d.fy = event.y;
    })
    .on("end", (event, d) => {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    });
}

/** Setup tick handler for simulation */
function setupTickHandler(
  simulation: d3.Simulation<MeshNode, MeshEdge>,
  edges: d3.Selection<SVGLineElement, MeshEdge, SVGGElement, unknown>,
  nodes: d3.Selection<SVGGElement, MeshNode, SVGGElement, unknown>
): void {
  simulation.on("tick", () => {
    requestAnimationFrame(() => {
      edges
        .attr("x1", d => (d.source as MeshNode).x!)
        .attr("y1", d => (d.source as MeshNode).y!)
        .attr("x2", d => (d.target as MeshNode).x!)
        .attr("y2", d => (d.target as MeshNode).y!);
      nodes.attr("transform", d => `translate(${d.x},${d.y})`);
    });
  });
}

// =============================================================================
// COMPONENT (Rule #4: Main render < 60 lines)
// =============================================================================

export const KnowledgeMesh: React.FC<KnowledgeMeshProps> = ({
  data,
  width = 800,
  height = 600,
  onNodeClick,
  onNodeHover,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const getNeighborIds = useCallback(
    (nodeId: string) => buildNeighborIds(nodeId, data.edges),
    [data.edges]
  );

  // Main D3 rendering effect
  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    const container = svg.append("g").attr("class", "mesh-container");

    const zoom = setupZoom(svg, container);
    const simNodes = data.nodes.map(n => ({ ...n }));
    const simEdges = data.edges.map(e => ({ ...e }));
    const simulation = createSimulation(simNodes, simEdges, width, height);

    const edges = renderEdges(container, simEdges);
    const nodes = renderNodes(container, simNodes, simulation);

    setupNodeHandlers(nodes, edges, svg, zoom, {
      width, height, data, getNeighborIds,
      selectedNodeId, setSelectedNodeId, onNodeClick, onNodeHover,
    });

    setupTickHandler(simulation, edges, nodes);
    setupBackgroundClick(svg, nodes, edges, setSelectedNodeId);

    return () => { simulation.stop(); };
  }, [data, width, height, getNeighborIds, onNodeClick, onNodeHover, selectedNodeId]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="w-full h-full bg-gray-900 rounded-lg cursor-grab active:cursor-grabbing"
      style={{ touchAction: 'none' }}
      data-testid="knowledge-mesh"
    />
  );
};

// =============================================================================
// EVENT HANDLERS (Rule #4: < 60 lines each)
// =============================================================================

interface NodeHandlerContext {
  width: number;
  height: number;
  data: MeshData;
  getNeighborIds: (nodeId: string) => Set<string>;
  selectedNodeId: string | null;
  setSelectedNodeId: (id: string | null) => void;
  onNodeClick?: (node: MeshNode, neighbors: MeshNode[]) => void;
  onNodeHover?: (node: MeshNode | null) => void;
}

/** Setup click, dblclick, hover handlers on nodes */
function setupNodeHandlers(
  nodes: d3.Selection<SVGGElement, MeshNode, SVGGElement, unknown>,
  edges: d3.Selection<SVGLineElement, MeshEdge, SVGGElement, unknown>,
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  zoom: d3.ZoomBehavior<SVGSVGElement, unknown>,
  ctx: NodeHandlerContext
): void {
  // Click handler
  nodes.on("click", (event, d) => {
    event.stopPropagation();
    handleNodeClick(d, nodes, edges, ctx);
  });

  // Double-click to center
  nodes.on("dblclick", (event, d) => {
    event.stopPropagation();
    centerOnNode(svg, zoom, d, ctx.width, ctx.height);
  });

  // Hover handlers
  nodes.on("mouseenter", (event, d) => {
    if (ctx.onNodeHover) ctx.onNodeHover(d);
    d3.select(event.currentTarget).select("circle")
      .transition().duration(100).attr("r", NODE_HOVER_RADIUS);
  });

  nodes.on("mouseleave", (event) => {
    if (ctx.onNodeHover) ctx.onNodeHover(null);
    d3.select(event.currentTarget).select("circle")
      .transition().duration(100).attr("r", NODE_RADIUS);
  });
}

/** Handle node click for neighbor highlighting */
function handleNodeClick(
  d: MeshNode,
  nodes: d3.Selection<SVGGElement, MeshNode, SVGGElement, unknown>,
  edges: d3.Selection<SVGLineElement, MeshEdge, SVGGElement, unknown>,
  ctx: NodeHandlerContext
): void {
  const neighborIds = ctx.getNeighborIds(d.id);
  const isAlreadySelected = ctx.selectedNodeId === d.id;

  if (isAlreadySelected) {
    resetHighlighting(nodes, edges, ctx.setSelectedNodeId);
  } else {
    applyHighlighting(d.id, neighborIds, nodes, edges, ctx.setSelectedNodeId);
  }

  if (ctx.onNodeClick) {
    const neighborNodes = ctx.data.nodes.filter(n => neighborIds.has(n.id));
    ctx.onNodeClick(d, neighborNodes);
  }
}

/** Reset all node/edge highlighting */
function resetHighlighting(
  nodes: d3.Selection<SVGGElement, MeshNode, SVGGElement, unknown>,
  edges: d3.Selection<SVGLineElement, MeshEdge, SVGGElement, unknown>,
  setSelectedNodeId: (id: string | null) => void
): void {
  setSelectedNodeId(null);
  nodes.select("circle").attr("opacity", FULL_OPACITY);
  nodes.select("text").attr("opacity", FULL_OPACITY);
  edges.attr("stroke-opacity", 0.4);
}

/** Apply highlighting to selected node and neighbors */
function applyHighlighting(
  nodeId: string,
  neighborIds: Set<string>,
  nodes: d3.Selection<SVGGElement, MeshNode, SVGGElement, unknown>,
  edges: d3.Selection<SVGLineElement, MeshEdge, SVGGElement, unknown>,
  setSelectedNodeId: (id: string | null) => void
): void {
  setSelectedNodeId(nodeId);

  nodes.select("circle").attr("opacity", (n: MeshNode) =>
    n.id === nodeId || neighborIds.has(n.id) ? FULL_OPACITY : DIM_OPACITY
  );
  nodes.select("text").attr("opacity", (n: MeshNode) =>
    n.id === nodeId || neighborIds.has(n.id) ? FULL_OPACITY : DIM_OPACITY
  );

  edges.attr("stroke-opacity", (e: MeshEdge) => {
    const sourceId = typeof e.source === 'string' ? e.source : e.source.id;
    const targetId = typeof e.target === 'string' ? e.target : e.target.id;
    return (sourceId === nodeId || targetId === nodeId)
      ? LINK_HIGHLIGHT_OPACITY : LINK_DIM_OPACITY;
  });
}

/** Center view on a node with smooth transition */
function centerOnNode(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  zoom: d3.ZoomBehavior<SVGSVGElement, unknown>,
  d: MeshNode,
  width: number,
  height: number
): void {
  const scale = 1.5;
  const x = width / 2 - (d.x || 0) * scale;
  const y = height / 2 - (d.y || 0) * scale;

  svg.transition()
    .duration(500)
    .call(zoom.transform as any, d3.zoomIdentity.translate(x, y).scale(scale));
}

/** Setup background click to deselect */
function setupBackgroundClick(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  nodes: d3.Selection<SVGGElement, MeshNode, SVGGElement, unknown>,
  edges: d3.Selection<SVGLineElement, MeshEdge, SVGGElement, unknown>,
  setSelectedNodeId: (id: string | null) => void
): void {
  svg.on("click", () => {
    resetHighlighting(nodes, edges, setSelectedNodeId);
  });
}

// =============================================================================
// HELPER: Map IFEntityArtifact to MeshData
// =============================================================================

export interface IFEntityArtifact {
  artifact_id: string;
  nodes: Array<{
    id: string;
    label: string;
    entity_type: string;
    metadata?: Record<string, unknown>;
  }>;
  edges: Array<{
    source_id: string;
    target_id: string;
    relationship: string;
    weight?: number;
  }>;
}

/** Convert IFEntityArtifact to MeshData format */
export function mapEntityArtifactToMesh(artifact: IFEntityArtifact): MeshData {
  const nodes: MeshNode[] = artifact.nodes.map(n => ({
    id: n.id,
    label: n.label,
    type: (n.entity_type?.toLowerCase() as EntityType) || 'unknown',
    metadata: n.metadata,
  }));

  const edges: MeshEdge[] = artifact.edges.map(e => ({
    source: e.source_id,
    target: e.target_id,
    label: e.relationship,
    weight: e.weight,
  }));

  return { nodes, edges };
}

export default KnowledgeMesh;
