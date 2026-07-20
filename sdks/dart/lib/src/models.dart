/// Typed models for the Forgefy Developer API.
///
/// Extracted payloads (features, questions, …) are model-generated: the known
/// fields are typed, and the full JSON object is always kept on `.raw` so any
/// extra fields the model emits remain accessible.
library;

/// Which extractors to run. Omit to run all four.
enum Extractor {
  features('features'),
  questions('questions'),
  conflicts('conflicts'),
  actionItems('action_items');

  const Extractor(this.wire);

  /// The value sent on the wire (`action_items`, not `actionItems`).
  final String wire;
}

/// Extraction quality tier.
enum ModelTier {
  standard('standard'),
  economy('economy');

  const ModelTier(this.wire);

  final String wire;
}

/// Lifecycle of an async extraction job.
enum JobStatus {
  queued,
  processing,
  done,
  failed,

  /// An unrecognized status string (forward-compatibility guard).
  unknown;

  static JobStatus fromWire(String? value) {
    switch (value) {
      case 'queued':
        return JobStatus.queued;
      case 'processing':
        return JobStatus.processing;
      case 'done':
        return JobStatus.done;
      case 'failed':
        return JobStatus.failed;
      default:
        return JobStatus.unknown;
    }
  }
}

/// A product feature inferred from the transcript.
class Feature {
  Feature.fromJson(Map<String, dynamic> json)
      : title = json['title'] as String? ?? '',
        description = json['description'] as String? ?? '',
        priority = json['priority'] as String? ?? '',
        raw = json;

  final String title;
  final String description;

  /// Usually one of `high` / `med` / `low`, but not guaranteed.
  final String priority;

  /// The complete JSON object, including any fields not typed above.
  final Map<String, dynamic> raw;
}

/// An open question raised in the transcript.
class Question {
  Question.fromJson(Map<String, dynamic> json)
      : text = json['text'] as String?,
        context = json['context'] as String?,
        raw = json;

  final String? text;
  final String? context;
  final Map<String, dynamic> raw;
}

/// A disagreement or contradiction detected in the transcript.
class Conflict {
  Conflict.fromJson(Map<String, dynamic> json)
      : description = json['description'] as String?,
        sideA = json['side_a'] as String?,
        sideB = json['side_b'] as String?,
        raw = json;

  final String? description;
  final String? sideA;
  final String? sideB;
  final Map<String, dynamic> raw;
}

/// A follow-up task extracted from the transcript.
class ActionItem {
  ActionItem.fromJson(Map<String, dynamic> json)
      : task = json['task'] as String?,
        owner = json['owner'] as String?,
        due = json['due'] as String?,
        raw = json;

  final String? task;
  final String? owner;
  final String? due;
  final Map<String, dynamic> raw;
}

/// Token consumption for a request.
class Usage {
  Usage.fromJson(Map<String, dynamic> json)
      : inputTokens = (json['input_tokens'] as num?)?.toInt() ?? 0,
        outputTokens = (json['output_tokens'] as num?)?.toInt() ?? 0;

  final int inputTokens;
  final int outputTokens;
}

/// Result of a synchronous `extract()` call.
class ExtractResult {
  ExtractResult.fromJson(Map<String, dynamic> json)
      : id = json['id'] as String? ?? '',
        modelTier = json['model_tier'] as String? ?? '',
        features =
            _mapList(json['features']).map(Feature.fromJson).toList(growable: false),
        questions =
            _mapList(json['questions']).map(Question.fromJson).toList(growable: false),
        conflicts =
            _mapList(json['conflicts']).map(Conflict.fromJson).toList(growable: false),
        actionItems = _mapList(json['action_items'])
            .map(ActionItem.fromJson)
            .toList(growable: false),
        usage = Usage.fromJson(
            (json['usage'] as Map?)?.cast<String, dynamic>() ?? const {});

  final String id;

  /// Which tier actually served the request — paid accounts over budget fall
  /// back to `economy`.
  final String modelTier;
  final List<Feature> features;
  final List<Question> questions;
  final List<Conflict> conflicts;
  final List<ActionItem> actionItems;
  final Usage usage;
}

/// Returned by `jobs.create()`.
class JobCreated {
  JobCreated.fromJson(Map<String, dynamic> json)
      : jobId = json['job_id'] as String? ?? '',
        status = JobStatus.fromWire(json['status'] as String?),
        webhookSecret = json['webhook_secret'] as String?;

  final String jobId;
  final JobStatus status;

  /// HMAC key for verifying webhook deliveries. Returned only at creation.
  final String? webhookSecret;
}

/// Returned by `jobs.get()` / `jobs.waitFor()`.
class JobStatusResponse {
  JobStatusResponse.fromJson(Map<String, dynamic> json)
      : jobId = json['job_id'] as String? ?? '',
        status = JobStatus.fromWire(json['status'] as String?),
        modelTier = json['model_tier'] as String? ?? '',
        createdAt = DateTime.tryParse(json['created_at'] as String? ?? ''),
        result = (json['result'] as Map?)?.cast<String, dynamic>(),
        error = json['error'] as String?;

  final String jobId;
  final JobStatus status;
  final String modelTier;
  final DateTime? createdAt;

  /// The extraction groups + usage, set once [status] is [JobStatus.done].
  final Map<String, dynamic>? result;

  /// Failure reason, set once [status] is [JobStatus.failed].
  final String? error;
}

/// The key owner's quota, from `usage()`.
class UsageResponse {
  UsageResponse.fromJson(Map<String, dynamic> json)
      : tier = json['tier'] as String? ?? '',
        tierName = json['tier_name'] as String? ?? '',
        monthlyTokens = (json['monthly_tokens'] as num?)?.toInt() ?? 0,
        tokensUsed = (json['tokens_used'] as num?)?.toInt() ?? 0,
        tokensRemaining = (json['tokens_remaining'] as num?)?.toInt() ?? 0,
        resetsAt = DateTime.tryParse(json['resets_at'] as String? ?? '');

  final String tier;
  final String tierName;
  final int monthlyTokens;
  final int tokensUsed;
  final int tokensRemaining;
  final DateTime? resetsAt;
}

List<Map<String, dynamic>> _mapList(Object? value) {
  if (value is List) {
    return value
        .whereType<Map>()
        .map((e) => e.cast<String, dynamic>())
        .toList(growable: false);
  }
  return const [];
}
