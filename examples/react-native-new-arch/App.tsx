import React, {useState} from 'react';
import {Button, SafeAreaView, StyleSheet, Text} from 'react-native';
import {
  detectText,
  initialize,
  type InitializeConfig,
  type TextResult,
} from '@rustors/react-native';

const config: InitializeConfig = {preset: 'ppv6'};

/**
 * Compile-only consumer sample. It never starts OCR automatically: model files
 * and a real user-selected image are application responsibilities.
 */
export default function App(): React.ReactElement {
  const [status, setStatus] = useState('Ready');

  const verifyApiShape = async (): Promise<void> => {
    await initialize(config);

    // Keep canonical source/output overloads checked by TypeScript.
    const lines: TextResult[] = await detectText({uri: '/tmp/invoice.png'})
      .asLines()
      .run();
    const words: TextResult[] = await detectText({base64: 'iVBORw0KGgo='})
      .withPreRecognition({lineYThreshold: 0.5, wordXThreshold: 0.4})
      .asWords()
      .run();
    const spatial: string = await detectText({
      bytes: new Uint8Array([0x89, 0x50, 0x4e, 0x47]),
    })
      .asSpatial()
      .run();

    setStatus(
      `API verified: ${lines.length + words.length} items, ${spatial.length} chars`,
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text testID="status">{status}</Text>
      <Button title="Verify RustO API" onPress={verifyApiShape} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12},
});
