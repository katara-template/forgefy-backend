/** Wire types for the Forgefy Developer API (responses keep API snake_case). */

export type Extractor = "features" | "questions" | "conflicts" | "action_items";
export type ModelTier = "standard" | "economy";
export type JobStatus = "queued" | "processing" | "done" | "failed";

/** Extracted payloads are model-generated; known fields typed, extras allowed. */
export interface Feature {
  title: string;
  description: string;
  priority: "high" | "med" | "low" | string;
  [key: string]: unknown;
}

export interface Question {
  text?: string;
  context?: string;
  [key: string]: unknown;
}

export interface Conflict {
  description?: string;
  side_a?: string;
  side_b?: string;
  [key: string]: unknown;
}

export interface ActionItem {
  task?: string;
  owner?: string | null;
  due?: string | null;
  [key: string]: unknown;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
}

export interface ExtractResult {
  id: string;
  /** Which tier actually served the request — paid accounts over budget fall back to "economy". */
  model_tier: ModelTier | string;
  features: Feature[];
  questions: Question[];
  conflicts: Conflict[];
  action_items: ActionItem[];
  usage: Usage;
}

export interface ExtractParams {
  transcript: string;
  /** Subset of extractors to run (you're only metered for these). Omit for all four. */
  extractors?: Extractor[];
  modelTier?: ModelTier;
}

export interface CreateJobParams extends ExtractParams {
  /** https URL POSTed the result when the job finishes (signed, see webhooks.verifySignature). */
  webhookUrl?: string;
  /** Dedupe key — reusing one returns the existing job instead of running a second. Auto-generated when omitted. */
  idempotencyKey?: string;
}

export interface JobCreated {
  job_id: string;
  status: JobStatus;
  /** HMAC key for verifying webhook deliveries. Returned only at creation. */
  webhook_secret: string | null;
}

export interface JobResult {
  [key: string]: unknown;
  usage?: Usage;
  features?: Feature[];
  questions?: Question[];
  conflicts?: Conflict[];
  action_items?: ActionItem[];
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  model_tier: ModelTier | string;
  created_at: string;
  result: JobResult | null;
  error: string | null;
}

export interface UsageResponse {
  tier: string;
  tier_name: string;
  monthly_tokens: number;
  tokens_used: number;
  tokens_remaining: number;
  resets_at: string;
}
