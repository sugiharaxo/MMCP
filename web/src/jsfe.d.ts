import "solid-js";

declare module "solid-js" {
  namespace JSX {
    interface IntrinsicElements {
      "jsfe-form": {
        value?: string;
        readonly?: boolean;
        schema?: any;
        [key: string]: any;
      };
    }
  }
}
