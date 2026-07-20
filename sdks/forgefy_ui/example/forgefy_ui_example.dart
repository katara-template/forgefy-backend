import 'package:flutter/material.dart' hide Spacer, Wrap;
import 'package:forgefy_ui/forgefy_ui.dart';

void main() => runApp(const ExampleApp());

class ExampleApp extends StatelessWidget {
  const ExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'forgefy_ui example',
      home: Scaffold(
        appBar: AppBar(title: const Text('forgefy_ui')),
        // A scrolling column with even spacing — no manual SizedBoxes.
        body: Scroll(
          padding: const EdgeInsets.all(16),
          spacing: 16,
          children: [
            // Header row: title on the left, action pushed to the right.
            const HStack(
              children: [
                Text('Dashboard', style: TextStyle(fontSize: 22)),
                Spacer(),
                Icon(Icons.settings),
              ],
            ),

            // Tags that wrap to the next line as needed.
            const Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                Chip(label: Text('design')),
                Chip(label: Text('flutter')),
                Chip(label: Text('layout')),
                Chip(label: Text('animation')),
              ],
            ),

            // Responsive column count, animated in one card at a time.
            Builder(
              builder: (context) {
                final columns = Responsive.value(context, mobile: 2, tablet: 3, desktop: 4);
                return Stagger(
                  spacing: 12,
                  children: [
                    Grid(
                      columns: columns,
                      spacing: 12,
                      runSpacing: 12,
                      children: List.generate(
                        6,
                        (i) => ScaleIn(
                          delay: Duration(milliseconds: 60 * i),
                          child: Card(
                            child: SizedBox(
                              height: 96,
                              child: Center(child: Text('Item $i')),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                );
              },
            ),

            const FadeIn(
              delay: Duration(milliseconds: 200),
              child: Text('Loaded.', textAlign: TextAlign.center),
            ),
          ],
        ),
      ),
    );
  }
}
