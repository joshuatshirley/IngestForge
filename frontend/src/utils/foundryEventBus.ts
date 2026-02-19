/**
 * US-1401.2.2: Foundry Event Bus
 * Centralized Pub/Sub for cross-pane synchronization.
 * Rule #4: Small scope, atomic handlers.
 * Rule #9: Type-safe event payloads.
 */

export type FoundryEventType = 
  | 'NODE_FOCUS' 
  | 'BBOX_SCROLL' 
  | 'FACT_VERIFIED' 
  | 'PIPELINE_COMMAND';

export interface FoundryEvent<T = any> {
  type: FoundryEventType;
  payload: T;
  timestamp: number;
}

type Handler = (event: FoundryEvent) => void;

const MAX_SUBSCRIBERS_PER_TYPE = 100;

class FoundryEventBus {
  private handlers: Map<FoundryEventType, Set<Handler>> = new Map();

  subscribe(type: FoundryEventType, handler: Handler) {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    
    const typeHandlers = this.handlers.get(type)!;
    
    // JPL Rule #2: Bound collection size
    if (typeHandlers.size >= MAX_SUBSCRIBERS_PER_TYPE) {
      console.warn(`Max subscribers reached for ${type}. Possible memory leak.`);
      return () => {};
    }

    typeHandlers.add(handler);
    return () => this.unsubscribe(type, handler);
  }

  unsubscribe(type: FoundryEventType, handler: Handler) {
    this.handlers.get(type)?.delete(handler);
  }

  emit(type: FoundryEventType, payload: any) {
    const event: FoundryEvent = {
      type,
      payload,
      timestamp: Date.now(),
    };
    this.handlers.get(type)?.forEach(handler => handler(event));
  }
}

export const eventBus = new FoundryEventBus();
