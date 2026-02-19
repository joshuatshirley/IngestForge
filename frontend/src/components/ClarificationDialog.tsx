/**
 * Query Clarification Dialog Component
 *
 * US-602: Modal dialog that prompts users to refine ambiguous queries
 * before execution, reducing null results from 40% → 20%.
 *
 * Features:
 * - Clarity score visualization (0-100%)
 * - Radio button selection of refinement suggestions
 * - "Use Original" / "Refine Query" actions
 * - API integration with /v1/query/clarify and /v1/query/refine
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  LinearProgress,
  Paper,
  RadioGroup,
  FormControlLabel,
  Radio,
  Alert,
  Chip,
} from '@mui/material';
import {
  CheckCircleOutline,
  WarningAmberOutlined,
  InfoOutlined,
} from '@mui/icons-material';

// =============================================================================
// Type Definitions
// =============================================================================

/**
 * Clarity score breakdown by factor.
 */
interface ClarityFactors {
  length: number;
  vagueness: number;
  specificity: number;
  word_count: number;
  question_structure: number;
}

/**
 * Response from POST /v1/query/clarify endpoint.
 */
export interface ClarifyQueryResponse {
  original_query: string;
  clarity_score: number; // 0.0 to 1.0
  is_clear: boolean;
  needs_clarification: boolean;
  suggestions: string[]; // Max 5
  reason: string;
  factors: ClarityFactors;
  evaluation_time_ms: number;
}

/**
 * Response from POST /v1/query/refine endpoint.
 */
interface RefineQueryResponse {
  refined_query: string;
  clarity_score: number;
  improvement: number;
  is_clear: boolean;
}

// =============================================================================
// Component Props
// =============================================================================

export interface ClarificationDialogProps {
  /** Whether the dialog is open */
  open: boolean;
  /** Original user query */
  originalQuery: string;
  /** Clarification data from API */
  clarificationData: ClarifyQueryResponse;
  /** Callback when dialog is closed */
  onClose: () => void;
  /** Callback when user refines query (receives refined query text) */
  onRefine: (refinedQuery: string) => void;
  /** Callback when user chooses to use original query */
  onUseOriginal: () => void;
  /** Optional API base URL (default: empty string for relative paths) */
  apiBaseUrl?: string;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Get color for clarity score.
 *
 * @param score - Clarity score (0.0 to 1.0)
 * @returns MUI color name
 */
function getClarityColor(score: number): 'error' | 'warning' | 'success' {
  if (score < 0.5) return 'error';
  if (score < 0.7) return 'warning';
  return 'success';
}

/**
 * Get icon for clarity score.
 *
 * @param score - Clarity score (0.0 to 1.0)
 * @returns React icon component
 */
function getClarityIcon(score: number): React.ReactElement {
  if (score < 0.5) return <WarningAmberOutlined color="error" />;
  if (score < 0.7) return <InfoOutlined color="warning" />;
  return <CheckCircleOutline color="success" />;
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * ClarificationDialog Component
 *
 * Displays query clarification suggestions and allows user to refine or proceed.
 *
 * @param props - Component props
 * @returns JSX.Element
 */
export const ClarificationDialog: React.FC<ClarificationDialogProps> = ({
  open,
  originalQuery,
  clarificationData,
  onClose,
  onRefine,
  onUseOriginal,
  apiBaseUrl = '',
}) => {
  // State
  const [selectedRefinement, setSelectedRefinement] = useState<string>('');
  const [isRefining, setIsRefining] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  // Destructure clarification data
  const {
    clarity_score,
    is_clear,
    suggestions,
    reason,
    evaluation_time_ms,
  } = clarificationData;

  // Calculate percentage for display
  const clarityPercentage = Math.round(clarity_score * 100);

  /**
   * Handle refine button click.
   * Calls POST /v1/query/refine API and invokes onRefine callback.
   */
  const handleRefine = async (): Promise<void> => {
    if (!selectedRefinement) {
      setError('Please select a refinement option');
      return;
    }

    setIsRefining(true);
    setError('');

    try {
      const response = await fetch(`${apiBaseUrl}/v1/query/refine`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          original_query: originalQuery,
          selected_refinement: selectedRefinement,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to refine query');
      }

      const data: RefineQueryResponse = await response.json();

      // Invoke callback with refined query
      onRefine(data.refined_query);

      // Reset state
      setSelectedRefinement('');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Refinement failed: ${errorMessage}`);
    } finally {
      setIsRefining(false);
    }
  };

  /**
   * Handle "Use Original" button click.
   * Invokes onUseOriginal callback and resets state.
   */
  const handleUseOriginal = (): void => {
    setSelectedRefinement('');
    setError('');
    onUseOriginal();
  };

  /**
   * Handle dialog close.
   * Resets state and invokes onClose callback.
   */
  const handleClose = (): void => {
    setSelectedRefinement('');
    setError('');
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="md"
      fullWidth
      aria-labelledby="clarification-dialog-title"
    >
      <DialogTitle id="clarification-dialog-title">
        <Box display="flex" alignItems="center" gap={1}>
          {getClarityIcon(clarity_score)}
          <Typography variant="h6">Clarify Your Query</Typography>
        </Box>
      </DialogTitle>

      <DialogContent>
        {/* Clarity Score Indicator */}
        <Box mb={3}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="body2" color="text.secondary">
              Clarity Score
            </Typography>
            <Chip
              label={`${clarityPercentage}%`}
              color={getClarityColor(clarity_score)}
              size="small"
            />
          </Box>
          <LinearProgress
            variant="determinate"
            value={clarityPercentage}
            color={getClarityColor(clarity_score)}
            sx={{ height: 8, borderRadius: 1 }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
            {reason}
          </Typography>
        </Box>

        {/* Original Query Display */}
        <Paper
          elevation={0}
          sx={{
            p: 2,
            mb: 3,
            bgcolor: 'grey.100',
            borderLeft: 3,
            borderColor: getClarityColor(clarity_score) + '.main',
          }}
        >
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Original Query:
          </Typography>
          <Typography variant="body1" fontWeight={500}>
            "{originalQuery}"
          </Typography>
        </Paper>

        {/* Suggestions */}
        {suggestions.length > 0 && (
          <Box mb={2}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              To improve your results, consider refining your query:
            </Typography>
            <RadioGroup
              value={selectedRefinement}
              onChange={(e) => setSelectedRefinement(e.target.value)}
            >
              {suggestions.map((suggestion, idx) => (
                <FormControlLabel
                  key={idx}
                  value={suggestion}
                  control={<Radio />}
                  label={
                    <Typography variant="body2">
                      {suggestion}
                    </Typography>
                  }
                  sx={{ mb: 1 }}
                />
              ))}
            </RadioGroup>
          </Box>
        )}

        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {/* Evaluation Time (Debug Info) */}
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
          Evaluated in {Math.round(evaluation_time_ms)}ms
        </Typography>
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button
          onClick={handleUseOriginal}
          color="inherit"
          disabled={isRefining}
        >
          Use Original
        </Button>
        <Button
          onClick={handleRefine}
          variant="contained"
          disabled={!selectedRefinement || isRefining}
          color="primary"
        >
          {isRefining ? 'Refining...' : 'Refine Query'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

// =============================================================================
// Hook for Query Clarification
// =============================================================================

/**
 * Custom hook for query clarification workflow.
 *
 * Usage:
 * ```tsx
 * const { clarify, isOpen, clarificationData, refine, useOriginal, close } = useQueryClarification();
 *
 * // Call clarify when user types query
 * await clarify(userQuery);
 *
 * // If isOpen === true, render <ClarificationDialog />
 * ```
 */
export function useQueryClarification(apiBaseUrl: string = '') {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [originalQuery, setOriginalQuery] = useState<string>('');
  const [clarificationData, setClarificationData] = useState<ClarifyQueryResponse | null>(null);
  const [onRefineCallback, setOnRefineCallback] = useState<((query: string) => void) | null>(null);

  /**
   * Clarify a query.
   *
   * @param query - User query to evaluate
   * @param threshold - Clarity threshold (0.0-1.0, default 0.7)
   * @param onRefineSuccess - Optional callback when query is refined
   * @returns Promise resolving to true if clarification needed, false if clear
   */
  const clarify = async (
    query: string,
    threshold: number = 0.7,
    onRefineSuccess?: (refinedQuery: string) => void
  ): Promise<boolean> => {
    try {
      const response = await fetch(`${apiBaseUrl}/v1/query/clarify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          threshold,
          use_llm: false,
        }),
      });

      if (!response.ok) {
        throw new Error('Clarification request failed');
      }

      const data: ClarifyQueryResponse = await response.json();

      if (data.needs_clarification) {
        setOriginalQuery(query);
        setClarificationData(data);
        setIsOpen(true);
        if (onRefineSuccess) {
          setOnRefineCallback(() => onRefineSuccess);
        }
        return true;
      }

      return false;
    } catch (err) {
      console.error('Query clarification error:', err);
      return false;
    }
  };

  /**
   * Handle query refinement.
   */
  const refine = (refinedQuery: string): void => {
    setIsOpen(false);
    if (onRefineCallback) {
      onRefineCallback(refinedQuery);
    }
  };

  /**
   * Handle use original query.
   */
  const useOriginal = (): void => {
    setIsOpen(false);
    if (onRefineCallback) {
      onRefineCallback(originalQuery);
    }
  };

  /**
   * Close dialog without action.
   */
  const close = (): void => {
    setIsOpen(false);
  };

  return {
    clarify,
    isOpen,
    originalQuery,
    clarificationData,
    refine,
    useOriginal,
    close,
  };
}

export default ClarificationDialog;
