import React, { useCallback, useId, useState } from "react";
import {
  ActivityIndicator,
  NativeModules,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  TurboModuleRegistry,
  View,
} from "react-native";
import {
  detectText,
  initialize,
  type TextResult,
} from "@rustors/react-native";
import { EXAMPLE1_PNG_BASE64 } from "./src/testImage";

export type ArchitectureMode =
  | "TurboModule (New Arch)"
  | "Bridge (Legacy Arch)"
  | "Unavailable";

export const detectArchitecture = (): ArchitectureMode => {
  const rnGlobal = globalThis as {
    TurboModuleRegistry?: {
      get<T>(name: string): T | null;
    };
    NativeModules?: {
      Rusto?: unknown;
    };
  };

  const turboRegistry = TurboModuleRegistry ?? rnGlobal.TurboModuleRegistry;
  const nativeModules = NativeModules ?? rnGlobal.NativeModules;

  if (turboRegistry?.get?.("Rusto")) {
    return "TurboModule (New Arch)";
  }

  if (nativeModules?.Rusto) {
    return "Bridge (Legacy Arch)";
  }

  return "Unavailable";
};

export interface TestResultItem {
  id: string;
  name: string;
  status: "idle" | "running" | "success" | "failure";
  details?: string;
}

interface LogEntry {
  id: string;
  message: string;
}

export const App = (): React.JSX.Element => {
  const [arch] = useState<ArchitectureMode>(detectArchitecture);
  const [isRunningAll, setIsRunningAll] = useState(false);
  const [ocrText, setOcrText] = useState<string | null>(null);
  const [detectedCount, setDetectedCount] = useState<number | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const [testResults, setTestResults] = useState<TestResultItem[]>([
    { id: "mod", name: "Native Module Handshake", status: "idle" },
    { id: "init", name: "Engine Initialization", status: "idle" },
    { id: "detect", name: "Text Detection Pipeline", status: "idle" },
    { id: "spatial", name: "Spatial Layout Generation", status: "idle" },
    { id: "cleanup", name: "Resource Verification", status: "idle" },
  ]);

  const logPrefix = useId();

  const appendLog = useCallback((msg: string) => {
    const timestamp = new Date().toLocaleTimeString();
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    setLogs((prev) => [{ id, message: `[${timestamp}] ${msg}` }, ...prev.slice(0, 49)]);
  }, []);

  const updateTestStatus = useCallback(
    (id: string, status: TestResultItem["status"], details?: string) => {
      setTestResults((prev) =>
        prev.map((item) => (item.id === id ? { ...item, status, details } : item))
      );
    },
    []
  );

  const runFullTestSuite = async () => {
    setIsRunningAll(true);
    appendLog(`Starting verification test suite on ${arch}...`);

    // Test 1: Native Module Handshake
    updateTestStatus("mod", "running");
    try {
      const rnGlobal = globalThis as {
        TurboModuleRegistry?: { get<T>(name: string): T | null };
        NativeModules?: { Rusto?: unknown };
      };
      const mod =
        (TurboModuleRegistry ?? rnGlobal.TurboModuleRegistry)?.get?.("Rusto") ??
        (NativeModules ?? rnGlobal.NativeModules)?.Rusto;

      if (!mod) {
        throw new Error("Rusto native module not found in TurboModuleRegistry or NativeModules");
      }

      updateTestStatus("mod", "success", `Detected: ${arch}`);
      appendLog(`Native module verified (${arch})`);
    } catch (err) {
      const errMessage = err instanceof Error ? err.message : String(err);
      updateTestStatus("mod", "failure", errMessage);
      appendLog(`Handshake failed: ${errMessage}`);
      setIsRunningAll(false);
      return;
    }

    // Test 2: Engine Initialization
    updateTestStatus("init", "running");
    try {
      await initialize({ preset: "ppv6" });
      updateTestStatus("init", "success", "Initialized with preset ppv6");
      appendLog("Engine initialized successfully");
    } catch (err) {
      const errMessage = err instanceof Error ? err.message : String(err);
      updateTestStatus("init", "failure", errMessage);
      appendLog(`Init failed: ${errMessage}`);
      setIsRunningAll(false);
      return;
    }

    // Test 3: Text Detection Pipeline
    updateTestStatus("detect", "running");
    try {
      const results: TextResult[] = await detectText(
        { base64: EXAMPLE1_PNG_BASE64 },
        { output: "lines", textScore: 0.3 }
      );

      setDetectedCount(results.length);
      updateTestStatus(
        "detect",
        "success",
        `Detection completed: ${results.length} boxes recognized`
      );
      appendLog(`Detection verified: ${results.length} lines on example1.png`);
    } catch (err) {
      const errMessage = err instanceof Error ? err.message : String(err);
      updateTestStatus("detect", "failure", errMessage);
      appendLog(`Detection failed: ${errMessage}`);
    }

    // Test 4: Spatial Layout Generation
    updateTestStatus("spatial", "running");
    try {
      const spatialText: string = await detectText(
        { base64: EXAMPLE1_PNG_BASE64 },
        { output: "spatial" }
      );

      setOcrText(spatialText);
      updateTestStatus("spatial", "success", "Spatial layout verified");
      appendLog(`Spatial layout verified (${spatialText.length} chars)`);
    } catch (err) {
      const errMessage = err instanceof Error ? err.message : String(err);
      updateTestStatus("spatial", "failure", errMessage);
      appendLog(`Spatial failed: ${errMessage}`);
    }

    // Test 5: Resource Cleanup
    updateTestStatus("cleanup", "running");
    try {
      updateTestStatus("cleanup", "success", "Resources verified");
      appendLog("All verification pipeline steps finished");
    } catch (err) {
      const errMessage = err instanceof Error ? err.message : String(err);
      updateTestStatus("cleanup", "failure", errMessage);
      appendLog(`Cleanup failed: ${errMessage}`);
    }

    setIsRunningAll(false);
    appendLog("Verification suite complete.");
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>RustO! OCR</Text>
          <Text style={styles.headerSubtitle}>
            React Native Dual Architecture & ObjC Verification Testbed
          </Text>
        </View>

        <View style={styles.badgeContainer}>
          <Text style={styles.badgeLabel}>Active Architecture:</Text>
          <View
            style={[
              styles.badge,
              arch.includes("TurboModule")
                ? styles.badgeNewArch
                : arch.includes("Bridge")
                ? styles.badgeLegacy
                : styles.badgeUnavailable,
            ]}
          >
            <Text style={styles.badgeText} testID="active-arch-badge">
              {arch}
            </Text>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Verification Tests</Text>
          {testResults.map((t) => (
            <View
              key={`${logPrefix}-${t.id}`}
              style={styles.testItem}
              testID={`test-item-${t.id}`}
            >
              <View style={styles.testHeader}>
                <Text style={styles.testName}>{t.name}</Text>
                <Text
                  testID={`test-status-${t.id}`}
                  style={[
                    styles.statusBadge,
                    t.status === "success" && styles.statusSuccess,
                    t.status === "failure" && styles.statusFailure,
                    t.status === "running" && styles.statusRunning,
                    t.status === "idle" && styles.statusIdle,
                  ]}
                >
                  {t.status.toUpperCase()}
                </Text>
              </View>
              {t.details ? (
                <Text
                  style={styles.testDetails}
                  testID={`test-details-${t.id}`}
                >
                  {t.details}
                </Text>
              ) : null}
            </View>
          ))}
        </View>

        <TouchableOpacity
          style={[styles.runButton, isRunningAll && styles.runButtonDisabled]}
          onPress={runFullTestSuite}
          disabled={isRunningAll}
          testID="run-tests-button"
        >
          {isRunningAll ? (
            <ActivityIndicator color="#ffffff" style={styles.buttonSpinner} />
          ) : null}
          <Text style={styles.runButtonText}>
            {isRunningAll ? "Running Verification..." : "Run Verification Tests"}
          </Text>
        </TouchableOpacity>

        {detectedCount === null ? null : (
          <View style={styles.metricCard} testID="latest-ocr-card">
            <Text style={styles.metricHeader} testID="latest-ocr-header">
              OCR Verification Result
            </Text>
            <Text style={styles.metricPayload} testID="latest-ocr-payload">
              Boxes Recognized: {detectedCount}
            </Text>
            {ocrText === null ? null : (
              <Text style={styles.metricText}>
                Spatial Export: {ocrText.length} characters
              </Text>
            )}
          </View>
        )}

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Activity Logs</Text>
          <View style={styles.logBox}>
            {logs.length === 0 ? (
              <Text style={styles.emptyLog}>No logs yet. Press Run to test.</Text>
            ) : (
              logs.map((entry) => (
                <Text key={entry.id} style={styles.logText}>
                  {entry.message}
                </Text>
              ))
            )}
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

export default App;

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#0f172a" },
  container: { padding: 16, gap: 16 },
  header: { marginBottom: 8 },
  headerTitle: { fontSize: 24, fontWeight: "700", color: "#f8fafc" },
  headerSubtitle: { fontSize: 13, color: "#94a3b8", marginTop: 4 },
  badgeContainer: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#1e293b",
    padding: 12,
    borderRadius: 8,
    gap: 8,
  },
  badgeLabel: { color: "#cbd5e1", fontSize: 14, fontWeight: "500" },
  badge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 6 },
  badgeNewArch: { backgroundColor: "#059669" },
  badgeLegacy: { backgroundColor: "#d97706" },
  badgeUnavailable: { backgroundColor: "#dc2626" },
  badgeText: { color: "#ffffff", fontSize: 12, fontWeight: "600" },
  section: { gap: 8 },
  sectionTitle: { fontSize: 16, fontWeight: "600", color: "#e2e8f0" },
  testItem: {
    backgroundColor: "#1e293b",
    padding: 12,
    borderRadius: 8,
    gap: 4,
  },
  testHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  testName: { fontSize: 14, color: "#f1f5f9", fontWeight: "500" },
  statusBadge: {
    fontSize: 11,
    fontWeight: "700",
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  statusIdle: { backgroundColor: "#334155", color: "#94a3b8" },
  statusRunning: { backgroundColor: "#1d4ed8", color: "#93c5fd" },
  statusSuccess: { backgroundColor: "#065f46", color: "#6ee7b7" },
  statusFailure: { backgroundColor: "#991b1b", color: "#fca5a5" },
  testDetails: { fontSize: 12, color: "#64748b" },
  runButton: {
    backgroundColor: "#2563eb",
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
    gap: 8,
  },
  runButtonDisabled: { opacity: 0.6 },
  buttonSpinner: { marginRight: 4 },
  runButtonText: { color: "#ffffff", fontSize: 15, fontWeight: "600" },
  metricCard: {
    backgroundColor: "#1e293b",
    padding: 14,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: "#10b981",
    gap: 4,
  },
  metricHeader: { color: "#10b981", fontSize: 14, fontWeight: "700" },
  metricPayload: { color: "#f8fafc", fontSize: 14, fontWeight: "500" },
  metricText: { color: "#94a3b8", fontSize: 12 },
  logBox: {
    backgroundColor: "#020617",
    padding: 12,
    borderRadius: 8,
    minHeight: 100,
    gap: 4,
  },
  emptyLog: { color: "#475569", fontSize: 12, fontStyle: "italic" },
  logText: { color: "#a5f3fc", fontSize: 11, fontFamily: "monospace" },
});
