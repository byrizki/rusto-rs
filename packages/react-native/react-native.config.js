module.exports = {
  dependency: {
    platforms: {
      android: {
        sourceDir: 'android',
        packageImportPath: 'import com.byrizki.rusto.reactnative.RustoPackage;',
        packageInstance: 'new RustoPackage()',
      },
    },
  },
};
