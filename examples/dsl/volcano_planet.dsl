; Volcanic Planet with Dynamic Eruptions
; A planet with active volcanoes that periodically erupt

; Create the main planet body using a spring mesh
(define planet-particles
  (let ([n 100]
        [radius 6.371e6])  ; Earth-like radius
    (loop i 0 n
      (let ([angle (* 2 pi (/ i n))]
            [r (* radius (+ 0.95 (random 0.1)))])  ; Slightly irregular
        (particle
          :mass 1e20
          :pos [(* r (cos angle)) (* r (sin angle))]
          :vel [0 0]
          :temp 300)))))

; Connect particles with springs to form rigid body
(spring-mesh planet-particles
  :stiffness 1e10
  :damping 1000)

; Create the planet composite
(define planet (composite planet-particles))

; Define volcano locations (as specific particles)
(define volcano-sites
  [(nth planet-particles 10)
   (nth planet-particles 35)
   (nth planet-particles 60)
   (nth planet-particles 85)])

; Persistent volcanic activity script
(rule 'volcanic-activity
  :persistent true
  :update-interval 5.0  ; Check every 5 seconds
  
  ; For each volcano site
  (loop volcano volcano-sites
    (when (> (random 1.0) 0.7)  ; 30% chance per check
      ; Eruption!
      (let ([eruption-force (* 1e15 (random 0.5 1.5))]
            [lava-temp (+ 1200 (random 300))])
        
        ; Create lava particles
        (loop i 0 20
          (let ([angle (random 0 (* 2 pi))]
                [speed (sqrt (/ (* 2 eruption-force) 1e18))])
            (particle
              :mass 1e18
              :pos (measure 'position volcano)
              :vel [(* speed (cos angle)) (* speed (sin angle))]
              :temp lava-temp)))
        
        ; Heat up surrounding area
        (let ([nearby (find :center (measure 'position volcano) 
                           :radius 1e5)])
          (loop p nearby
            (set-temperature p (+ (measure 'temperature p) 100))))
        
        ; Log eruption
        (print (format "Volcano at ~a erupted with force ~a N" 
                      volcano eruption-force))))))

; Rule for lava cooling
(rule 'lava-cooling
  :persistent true
  :update-interval 0.1
  
  (let ([hot-particles (query :property 'temperature :min 1000)])
    (loop p hot-particles
      ; Cool down over time
      (set-temperature p (* (measure 'temperature p) 0.99))
      
      ; Solidify when cool enough
      (when (< (measure 'temperature p) 600)
        ; Find nearest planet particle and create spring
        (let ([nearest (find-nearest p planet-particles)])
          (spring p nearest :stiffness 1e8 :damping 100))))))

; Monitor volcanic activity
(watch
  :every 1.0
  (let ([avg-temp (measure 'average-temperature planet-particles)]
        [lava-count (length (query :property 'temperature :min 1000))])
    (print (format "Planet temp: ~a K, Active lava: ~a particles"
                  avg-temp lava-count))))