try:
    from diffusers import PixArtAlphaPipeline
    print("PixArtAlphaPipeline available")
except Exception as e:
    print("PixArtAlphaPipeline error:", e)

try:
    from diffusers import PixArtXLPipeline
    print("PixArtXLPipeline available")
except Exception as e:
    print("PixArtXLPipeline error:", e)

try:
    from diffusers import PixArtSigmaPipeline
    print("PixArtSigmaPipeline available")
except Exception as e:
    print("PixArtSigmaPipeline error:", e)
