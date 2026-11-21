import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// Base URL of the deployed API (no trailing slash).
/// The default below is the deployed Render URL. Override it at run time to use another API:
///   flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000
/// (10.0.2.2 is how the Android emulator reaches the computer's localhost.)
const String kApiBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'https://birthweight-api.onrender.com',
);

void main() => runApp(const BirthWeightApp());

class BirthWeightApp extends StatelessWidget {
  const BirthWeightApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Birth Weight Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const PredictPage(),
    );
  }
}

/// How a field's text is turned into JSON before it is sent to the API.
enum FieldKind { integer, decimal, text }

class FieldSpec {
  final String key; // must match the API field name
  final String label;
  final String helper;
  final FieldKind kind;
  const FieldSpec(this.key, this.label, this.helper, this.kind);
}

/// One text field per model input (9 inputs, same order as the API).
const List<FieldSpec> kFields = [
  FieldSpec('ga_weeks', 'Gestational age (weeks)', 'Number from 20 to 44, e.g. 38', FieldKind.decimal),
  FieldSpec('mothers_age', "Mother's age (years)", 'Whole number from 12 to 55, e.g. 27', FieldKind.integer),
  FieldSpec('multiple', 'Plurality', 'Whole number from 1 to 5 (1 = single baby, 2 = twins)', FieldKind.integer),
  FieldSpec('sex', "Baby's sex", 'Male, Female or Not_indicated', FieldKind.text),
  FieldSpec('country', 'Country', 'Uganda or Kenya', FieldKind.text),
  FieldSpec('facility', 'Facility', 'e.g. Kamuli_Hospital or Migori_County_Referral_Hospital', FieldKind.text),
  FieldSpec('mode_of_delivery', 'Mode of delivery',
      'Normal_Vaginal_Delivery, Caesarean_Section, Born_Before_Arrival or Other', FieldKind.text),
  FieldSpec('viability', 'Viability', 'No Indication, IUFD or Abortion', FieldKind.text),
  FieldSpec('facility_birth', 'Facility birth', 'Facility birth or BBA', FieldKind.text),
];

final Map<String, String> kLabels = {for (final f in kFields) f.key: f.label};

class PredictPage extends StatefulWidget {
  const PredictPage({super.key});

  @override
  State<PredictPage> createState() => _PredictPageState();
}

class _PredictPageState extends State<PredictPage> {
  final Map<String, TextEditingController> _controllers = {
    for (final f in kFields) f.key: TextEditingController(),
  };
  final ScrollController _scroll = ScrollController();

  bool _loading = false;
  String? _error; // readable error text, shown in the result area
  double? _kg; // predicted weight in kg
  bool _lowBirthWeight = false;

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    _scroll.dispose();
    super.dispose();
  }

  /// Builds the JSON body. Empty fields are left out so the API reports them as missing.
  /// Text that cannot be parsed as a number is sent as-is so the API reports the wrong type.
  Map<String, dynamic> _buildBody() {
    final body = <String, dynamic>{};
    for (final f in kFields) {
      final raw = _controllers[f.key]!.text.trim();
      if (raw.isEmpty) continue;
      switch (f.kind) {
        case FieldKind.integer:
          body[f.key] = int.tryParse(raw) ?? double.tryParse(raw) ?? raw;
        case FieldKind.decimal:
          body[f.key] = double.tryParse(raw) ?? raw;
        case FieldKind.text:
          body[f.key] = raw;
      }
    }
    return body;
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    setState(() {
      _loading = true;
      _error = null;
      _kg = null;
    });

    try {
      final response = await http
          .post(
            Uri.parse('$kApiBaseUrl/predict'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(_buildBody()),
          )
          .timeout(const Duration(seconds: 90));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _kg = (data['predicted_birth_weight_kg'] as num).toDouble();
          _lowBirthWeight = data['low_birth_weight'] == true;
        });
      } else if (response.statusCode == 422) {
        setState(() => _error = _formatValidationErrors(response.body));
      } else {
        setState(() => _error = 'The server returned an error (${response.statusCode}). Please try again.');
      }
    } on TimeoutException {
      setState(() => _error =
          'The server took too long to answer. A free Render service can need up to a minute to wake up, so try again.');
    } on SocketException {
      setState(() => _error = 'Cannot reach the server. Check your internet connection and the API address.');
    } catch (e) {
      setState(() => _error = 'Something went wrong: $e');
    } finally {
      if (mounted) {
        setState(() => _loading = false);
        // scroll down so the result area is visible
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (_scroll.hasClients) {
            _scroll.animateTo(
              _scroll.position.maxScrollExtent,
              duration: const Duration(milliseconds: 300),
              curve: Curves.easeOut,
            );
          }
        });
      }
    }
  }

  /// Turns FastAPI's 422 JSON into one readable line per problem.
  String _formatValidationErrors(String body) {
    try {
      final decoded = jsonDecode(body) as Map<String, dynamic>;
      final detail = decoded['detail'];
      if (detail is List) {
        final lines = <String>[];
        for (final item in detail) {
          final loc = (item['loc'] as List).map((e) => e.toString()).toList();
          final key = loc.isNotEmpty ? loc.last : '';
          final label = kLabels[key] ?? key;
          lines.add('$label: ${_friendly(item['msg'].toString())}');
        }
        return lines.join('\n');
      }
      return detail.toString();
    } catch (_) {
      return 'The server rejected the input. Check every field and try again.';
    }
  }

  String _friendly(String msg) {
    if (msg == 'Field required') return 'this value is required.';
    if (msg.contains('fractional part')) return 'enter a whole number (no decimals).';
    if (msg.startsWith('Input should be a valid')) return 'wrong type, enter a valid number.';
    final ge = RegExp(r'greater than or equal to (.+)').firstMatch(msg);
    if (ge != null) return 'must be at least ${ge.group(1)}.';
    final le = RegExp(r'less than or equal to (.+)').firstMatch(msg);
    if (le != null) return 'must be at most ${le.group(1)}.';
    if (msg.startsWith("Input should be '")) {
      return 'not an allowed value. Use one of: ${msg.replaceFirst('Input should be ', '').replaceAll("'", '')}.';
    }
    return msg;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Birth Weight Predictor')),
      body: SafeArea(
        child: SingleChildScrollView(
          controller: _scroll,
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Enter the details from the maternity register to estimate the newborn\'s birth weight.',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 16),
              for (final f in kFields) ...[
                TextField(
                  controller: _controllers[f.key],
                  // plain text keyboard on purpose, so a wrong type can also be tried from the keyboard
                  keyboardType: TextInputType.text,
                  textInputAction: TextInputAction.next,
                  decoration: InputDecoration(
                    labelText: f.label,
                    helperText: f.helper,
                    helperMaxLines: 2,
                    border: const OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 12),
              ],
              const SizedBox(height: 4),
              SizedBox(
                height: 48,
                child: FilledButton(
                  onPressed: _loading ? null : _predict,
                  child: const Text('Predict'),
                ),
              ),
              const SizedBox(height: 16),
              _buildResultArea(context),
            ],
          ),
        ),
      ),
    );
  }

  /// The single display area: placeholder, loading, result or error.
  Widget _buildResultArea(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final textTheme = Theme.of(context).textTheme;

    Widget content;
    Color background;

    if (_loading) {
      background = scheme.surfaceContainerHighest;
      content = const Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2)),
          SizedBox(width: 12),
          Flexible(child: Text('Contacting the server...')),
        ],
      );
    } else if (_error != null) {
      background = scheme.errorContainer;
      content = Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.error_outline, color: scheme.onErrorContainer),
          const SizedBox(width: 12),
          Expanded(
            child: Text(_error!, style: textTheme.bodyMedium?.copyWith(color: scheme.onErrorContainer)),
          ),
        ],
      );
    } else if (_kg != null) {
      background = scheme.primaryContainer;
      content = Column(
        children: [
          Text('Predicted birth weight', style: textTheme.titleSmall?.copyWith(color: scheme.onPrimaryContainer)),
          const SizedBox(height: 4),
          Text(
            '${_kg!.toStringAsFixed(2)} kg',
            style: textTheme.headlineMedium?.copyWith(
              color: scheme.onPrimaryContainer,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            '(${(_kg! * 1000).round()} g)',
            style: textTheme.bodyMedium?.copyWith(color: scheme.onPrimaryContainer),
          ),
          const SizedBox(height: 8),
          Text(
            _lowBirthWeight
                ? 'Below 2.5 kg: likely low birth weight, prepare neonatal care.'
                : 'At or above 2.5 kg: not low birth weight.',
            textAlign: TextAlign.center,
            style: textTheme.bodyMedium?.copyWith(color: scheme.onPrimaryContainer),
          ),
        ],
      );
    } else {
      background = scheme.surfaceContainerHighest;
      content = Text(
        'The predicted birth weight will appear here.',
        textAlign: TextAlign.center,
        style: textTheme.bodyMedium,
      );
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: background, borderRadius: BorderRadius.circular(12)),
      child: content,
    );
  }
}
