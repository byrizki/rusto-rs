import React from "react";
import TestRenderer, { act } from "react-test-renderer";
import { App } from "./App";

declare global {
  var TurboModuleRegistry:
    | {
        get<T>(name: string): T | null;
      }
    | undefined;

  var NativeModules:
    | {
        Rusto?: {
          initialize: (config: unknown) => Promise<void>;
          detectText: (source: unknown, options?: unknown) => Promise<unknown>;
        };
      }
    | undefined;
}

describe("React Native Architecture & Bridging Verification", () => {
  const mockRustoModule = {
    initialize: jest.fn().mockResolvedValue(undefined),
    detectText: jest.fn().mockImplementation((source, options) => {
      if (options?.output === "spatial") {
        return Promise.resolve("SPATIAL TEXT LAYOUT");
      }
      return Promise.resolve([
        {
          text: "Sample Line",
          score: 0.95,
          box_points: [
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10],
          ],
          frame: { width: 10, height: 10, top: 0, left: 0 },
        },
      ]);
    }),
  };

  afterEach(() => {
    globalThis.TurboModuleRegistry = undefined;
    globalThis.NativeModules = undefined;
    jest.clearAllMocks();
  });

  it("verifies and renders on TurboModule (New Architecture)", async () => {
    globalThis.TurboModuleRegistry = {
      get: (name: string) => (name === "Rusto" ? mockRustoModule : null),
    };

    let renderer!: TestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = TestRenderer.create(<App />);
    });

    const badge = renderer.root.findByProps({ testID: "active-arch-badge" });
    expect(badge.props.children).toBe("TurboModule (New Arch)");

    const runButton = renderer.root.findByProps({ testID: "run-tests-button" });
    await act(async () => {
      await runButton.props.onPress();
    });

    expect(mockRustoModule.initialize).toHaveBeenCalledTimes(1);
    expect(mockRustoModule.detectText).toHaveBeenCalledTimes(2);

    const modStatus = renderer.root.findByProps({ testID: "test-status-mod" });
    expect(modStatus.props.children).toBe("SUCCESS");

    const initStatus = renderer.root.findByProps({ testID: "test-status-init" });
    expect(initStatus.props.children).toBe("SUCCESS");

    const detectStatus = renderer.root.findByProps({ testID: "test-status-detect" });
    expect(detectStatus.props.children).toBe("SUCCESS");

    const spatialStatus = renderer.root.findByProps({ testID: "test-status-spatial" });
    expect(spatialStatus.props.children).toBe("SUCCESS");

    const cleanupStatus = renderer.root.findByProps({ testID: "test-status-cleanup" });
    expect(cleanupStatus.props.children).toBe("SUCCESS");
  });

  it("verifies and renders on Bridge (Legacy Architecture)", async () => {
    globalThis.NativeModules = {
      Rusto: mockRustoModule,
    };

    let renderer!: TestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = TestRenderer.create(<App />);
    });

    const badge = renderer.root.findByProps({ testID: "active-arch-badge" });
    expect(badge.props.children).toBe("Bridge (Legacy Arch)");

    const runButton = renderer.root.findByProps({ testID: "run-tests-button" });
    await act(async () => {
      await runButton.props.onPress();
    });

    expect(mockRustoModule.initialize).toHaveBeenCalledTimes(1);
    expect(mockRustoModule.detectText).toHaveBeenCalledTimes(2);

    const detectStatus = renderer.root.findByProps({ testID: "test-status-detect" });
    expect(detectStatus.props.children).toBe("SUCCESS");
  });
});
