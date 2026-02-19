"use client";

import { useDispatch, useSelector } from 'react-redux';
import { 
  useChatMutation,
  useToggleBookmarkMutation, 
  useAddTagMutation 
} from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';
import { RootState } from '@/store';
import { setSearching, addMessage } from '@/store/slices/searchSlice';

/**
 * US-1401.2.1: Custom hook for research page actions.
 * Extracts logic to satisfy JPL Rule #4 (Function Length).
 */
export function useResearchActions() {
  const dispatch = useDispatch();
  const { showToast } = useToast();
  const { query, conversation } = useSelector((state: RootState) => state.search);

  const [chat, { isLoading: isChatting }] = useChatMutation();
  const [toggleBookmark] = useToggleBookmarkMutation();
  const [addTag] = useAddTagMutation();

  const handleSearch = async (
    customQuery?: string, 
    options?: { broadcast?: boolean, nexus_ids?: string[] }
  ) => {
    const searchTarget = customQuery || query;
    if (!searchTarget.trim()) return;

    dispatch(setSearching(true));
    dispatch(addMessage({ role: 'user', text: searchTarget }));

    try {
      const messagesPayload = [...conversation, { role: 'user', text: searchTarget }];
      const response = await chat({ 
        messages: messagesPayload,
        broadcast: options?.broadcast,
        nexus_ids: options?.nexus_ids
      }).unwrap();
      dispatch(addMessage({ 
        role: 'ai', 
        text: response.answer,
        results: response.sources 
      }));
    } catch (err) {
      showToast('Chat failed. Engine may be offline.', 'error');
    } finally {
      dispatch(setSearching(false));
    }
  };

  const onToggleBookmark = async (id: string) => {
    try {
      await toggleBookmark(id).unwrap();
      showToast('Bookmark updated', 'success');
    } catch (err) {
      showToast('Action failed', 'error');
    }
  };

  const onAddTag = async (id: string) => {
    const tag = prompt('Enter tag name:');
    if (!tag) return;
    try {
      await addTag({ chunkId: id, tag }).unwrap();
      showToast(`Added tag: ${tag}`, 'success');
    } catch (err) {
      showToast('Action failed', 'error');
    }
  };

  return {
    handleSearch,
    onToggleBookmark,
    onAddTag,
    isChatting
  };
}
