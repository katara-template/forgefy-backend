// ignore_for_file: avoid_print
import 'package:forgefy/forgefy.dart';

Future<void> main() async {
  final forgefy = Forgefy(
    apiKey: 'fgy_live_...', // from the dashboard's Developers page
    baseUrl: 'https://your-forgefy-host',
  );

  try {
    // Synchronous extraction (transcripts up to 50k chars).
    final result = await forgefy.extract(
      transcript: 'We need Google login before launch. Sarah owns billing.',
      extractors: [Extractor.features, Extractor.actionItems], // omit for all four
    );

    for (final f in result.features) {
      print('[${f.priority}] ${f.title} — ${f.description}');
    }
    print('tokens: ${result.usage.inputTokens} in / ${result.usage.outputTokens} out');

    // Async job for a long transcript, polled until done.
    final job = await forgefy.jobs.create(transcript: 'a very long transcript…');
    final done = await forgefy.jobs.waitFor(job.jobId);
    print('job ${done.jobId}: ${done.result}');
  } on QuotaExceededException catch (e) {
    print('Out of quota: ${e.detail}');
  } on ForgefyException catch (e) {
    print('API error ${e.status}: ${e.message}');
  } finally {
    forgefy.close();
  }
}
