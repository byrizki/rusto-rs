import React from 'react';
import renderer, { act } from 'react-test-renderer';
import App from '../App';

jest.mock('react-native-rusto', () => ({
  initialize: jest.fn(() => Promise.resolve()),
  detectText: jest.fn((source, options) =>
    Promise.resolve(options?.output === 'spatial' ? 'receipt total' : []),
  ),
}));

it('calls canonical RustO initialize and detectText overloads', async () => {
  const tree = renderer.create(<App />);
  await act(async () => {
    tree.root.findByProps({ title: 'Verify RustO API' }).props.onPress();
  });
  expect(tree.root.findByProps({ testID: 'status' }).props.children).toBe(
    'API verified: 0 items, 13 chars',
  );
});
