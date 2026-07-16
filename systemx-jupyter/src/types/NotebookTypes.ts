export interface ExtractedCell {
  cell_id: string;
  source: string;
  execution_count?: number | null;
  document_index?: number;
  is_dirty?: boolean | null;
}
