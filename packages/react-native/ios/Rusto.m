#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(Rusto, NSObject)

RCT_EXTERN_METHOD(initialize:(nullable NSString *)detModel
                  recModel:(nullable NSString *)recModel
                  dict:(nullable NSString *)dict
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(detectText:(NSString *)imagePath
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(detectTextFromBytes:(NSString *)imageData
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(getVersion:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

+ (BOOL)requiresMainQueueSetup
{
  return NO;
}

@end
