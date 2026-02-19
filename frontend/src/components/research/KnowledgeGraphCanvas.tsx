"use client";

import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useWorkbenchContext } from '../../context/WorkbenchContext';
import { eventBus } from '../../utils/foundryEventBus';

interface Node extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  type: string;
}

interface Link extends d3.SimulationLinkDatum<Node> {
  source: string | Node;
  target: string | Node;
  label: string;
}

interface GraphProps {
  data: {
    nodes: Node[];
    links: Link[];
  };
}

export const KnowledgeGraphCanvas = ({ data }: GraphProps) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const { activeNodeId, setActiveNode, setCurrentDocument } = useWorkbenchContext();

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const width = svgRef.current.clientWidth || 800;
    const height = svgRef.current.clientHeight || 600;

    // Clear previous SVG contents
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Create container for zoom support
    const g = svg.append("g");

    // Setup Zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));

    svg.call(zoom as any);

    // Force Simulation
    const simulation = d3.forceSimulation<Node>(data.nodes)
      .force("link", d3.forceLink<Node, Link>(data.links).id(d => d.id).distance(150))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(50));

    // Links
    const link = g.append("g")
      .attr("stroke", "#374151")
      .attr("stroke-opacity", 0.6)
      .selectAll("line")
      .data(data.links)
      .join("line")
      .attr("stroke-width", 1.5);

    // Nodes
    const node = g.append("g")
      .selectAll("g")
      .data(data.nodes)
      .join("g")
      .attr("cursor", "pointer")
      .on("click", (event, d) => {
        setActiveNode(d.id);
        
        // Emit event for other panes (Task 104)
        eventBus.emit('NODE_FOCUS', { id: d.id, type: d.type });

        if (d.type === 'document') {
          setCurrentDocument(d.id);
        }
      })
      .call(d3.drag<any, any>()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended) as any
      );

    // Node Circles
    node.append("circle")
      .attr("r", 12)
      .attr("fill", d => {
        if (d.id === activeNodeId) return '#00f2ff'; // Active highlight
        if (d.type === 'concept') return '#e94560';
        if (d.type === 'document') return '#4fc3f7';
        if (d.type === 'entity') return '#ffc107';
        return '#9c27b0';
      })
      .attr("stroke", d => d.id === activeNodeId ? '#fff' : "#1a1a2e")
      .attr("stroke-width", d => d.id === activeNodeId ? 3 : 2)
      .attr("class", "transition-all duration-300");

    // Node Labels
    node.append("text")
      .text(d => d.label)
      .attr("x", 16)
      .attr("y", 4)
      .attr("fill", d => d.id === activeNodeId ? '#fff' : "#94a3b8")
      .style("font-size", d => d.id === activeNodeId ? "12px" : "10px")
      .style("font-weight", "bold")
      .style("pointer-events", "none");

    simulation.on("tick", () => {
      link
        .attr("x1", d => (d.source as Node).x!)
        .attr("y1", d => (d.source as Node).y!)
        .attr("x2", d => (d.target as Node).x!)
        .attr("y2", d => (d.target as Node).y!);

      node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    return () => { simulation.stop(); };
  }, [data, activeNodeId, setActiveNode, setCurrentDocument]);

  return (
    <svg 
      ref={svgRef} 
      className="w-full h-full cursor-grab active:cursor-grabbing outline-none"
    />
  );
};
