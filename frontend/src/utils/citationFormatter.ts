/**
 * Citation Formatter Utility
 *
 * Provides standard academic formatting for search result citations.
 * Fulfills US-0203 AC for APA/MLA/Chicago support.
 *
 * JPL Compliance:
 * - Rule #4: Small, testable pure functions.
 * - Rule #9: 100% type hints.
 */

export interface CitationData {
  author?: string;
  sourceTitle: string;
  section?: string;
  pageStart?: number;
  pageEnd?: number;
  year?: string;
}

/**
 * Format citation in APA style.
 * Format: Author, A. A. (Year). Title. Section, pp. X-Y.
 */
export function formatAPA(data: CitationData): string {
  const author = data.author || 'Unknown Author';
  const year = data.year ? `(${data.year})` : '(n.d.)';
  const pages = data.pageStart ? `, pp. ${data.pageStart}${data.pageEnd ? `-${data.pageEnd}` : ''}` : '';
  const section = data.section ? `. ${data.section}` : '';
  
  return `${author} ${year}. ${data.sourceTitle}${section}${pages}.`;
}

/**
 * Format citation in MLA style.
 * Format: Author. "Section." Title, Year, pp. X-Y.
 */
export function formatMLA(data: CitationData): string {
  const author = data.author || 'Unknown Author';
  const section = data.section ? `"${data.section}." ` : '';
  const pages = data.pageStart ? `, pp. ${data.pageStart}${data.pageEnd ? `-${data.pageEnd}` : ''}` : '';
  const year = data.year ? `, ${data.year}` : '';
  
  return `${author}. ${section}${data.sourceTitle}${year}${pages}.`;
}

/**
 * Format citation in Chicago style.
 * Format: Author, "Section," in Title (Year), X-Y.
 */
export function formatChicago(data: CitationData): string {
  const author = data.author || 'Unknown Author';
  const section = data.section ? `"${data.section}," ` : '';
  const pages = data.pageStart ? `, ${data.pageStart}${data.pageEnd ? `-${data.pageEnd}` : ''}` : '';
  const year = data.year ? ` (${data.year})` : '';
  
  return `${author}, ${section}in ${data.sourceTitle}${year}${pages}.`;
}

export type CitationStyle = 'APA' | 'MLA' | 'Chicago';

export function formatCitation(data: CitationData, style: CitationStyle): string {
  switch (style) {
    case 'APA': return formatAPA(data);
    case 'MLA': return formatMLA(data);
    case 'Chicago': return formatChicago(data);
    default: return formatAPA(data);
  }
}
