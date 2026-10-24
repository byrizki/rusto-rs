const path = require("path");
const { getDefaultConfig, mergeConfig } = require("@react-native/metro-config");

const root = path.resolve(__dirname, "../../..");
const pak = path.resolve(__dirname, "..");
const react = path.resolve(__dirname, "node_modules/react");
const reactNative = path.resolve(__dirname, "node_modules/react-native");

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('metro-config').MetroConfig}
 */
const config = {
  watchFolders: [root, pak],
  resolver: {
    nodeModulesPaths: [
      path.resolve(__dirname, "node_modules"),
      path.resolve(pak, "node_modules"),
    ],
    extraNodeModules: {
      "@rustors/react-native": pak,
      react,
      "react-native": reactNative,
    },
  },
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
