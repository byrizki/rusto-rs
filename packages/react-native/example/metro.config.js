/**
 * Metro configuration for React Native
 * https://github.com/facebook/react-native
 *
 * @format
 */

module.exports = {
  transformer: {
    getTransformOptions: () =>
      Promise.resolve({
        transform: {
          experimentalImportSupport: false,
          inlineRequires: true,
        },
      }),
  },
};
