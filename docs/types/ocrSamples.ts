export interface SampleImage {
  id: string;
  name: string;
  description: string;
  url: string;
}

export function getSampleImages(baseURL: string): SampleImage[] {
  return [
    {
      id: 'invoice',
      name: 'Invoice',
      description: 'Structured layout with numbers & tables',
      url: `${baseURL}samples/example1.png`,
    },
    {
      id: 'idcard',
      name: 'Passport',
      description: 'Identity document with personal details & MRZ',
      url: `${baseURL}samples/idcard.jpg`,
    },
    {
      id: 'handwritten',
      name: 'Handwritten',
      description: 'Vintage cursive handwriting receipt',
      url: `${baseURL}samples/invoice1.jpg`,
    },
    {
      id: 'dense-newspaper',
      name: 'Dense Newspaper',
      description: 'Dense historical multi-column newspaper (ChronAm)',
      url: `${baseURL}samples/dense-newspaper.png`,
    },
  ];
}
