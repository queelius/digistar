; AI-Controlled Spaceship
; A spaceship that autonomously navigates, avoids obstacles, and seeks targets

; Create spaceship structure
(define ship-hull
  (let ([particles '()])
    ; Create hull shape (simplified as hexagon)
    (loop i 0 6
      (let ([angle (* i (/ (* 2 pi) 6))]
            [radius 50])
        (push particles
          (particle
            :mass 1000
            :pos [(* radius (cos angle)) (* radius (sin angle))]
            :vel [0 0]
            :temp 290))))
    
    ; Add center particle
    (push particles
      (particle :mass 5000 :pos [0 0] :vel [0 0] :temp 290))
    
    particles))

; Connect with springs for rigid structure
(spring-mesh ship-hull :stiffness 1e6 :damping 100)

; Create ship composite
(define ship (composite ship-hull))

; Ship state variables
(define ship-fuel 10000)
(define ship-health 100)
(define ship-target nil)
(define ship-mode 'patrol)  ; patrol, pursue, flee, dock

; Navigation AI - runs continuously
(rule 'ship-ai
  :persistent true
  :update-interval 0.1
  
  (let ([ship-pos (measure 'center-of-mass ship-hull)]
        [ship-vel (measure 'average-velocity ship-hull)])
    
    ; Obstacle avoidance (highest priority)
    (let ([nearby (find :center ship-pos :radius 500)])
      (when (> (length nearby) (length ship-hull))  ; Other objects nearby
        (let ([avoidance-vec [0 0]])
          (loop obj nearby
            (when (not (member obj ship-hull))
              (let ([obj-pos (measure 'position obj)]
                    [dist (distance ship-pos obj-pos)])
                (when (< dist 300)  ; Danger zone
                  ; Calculate repulsion vector
                  (let ([dx (- (first ship-pos) (first obj-pos))]
                        [dy (- (second ship-pos) (second obj-pos))]
                        [force (/ 10000 (* dist dist))])
                    (set avoidance-vec
                      [(+ (first avoidance-vec) (* force dx))
                       (+ (second avoidance-vec) (* force dy))])))))
          
          ; Apply avoidance thrust
          (when (> (magnitude avoidance-vec) 0.1)
            (apply-thrust ship avoidance-vec)
            (set ship-fuel (- ship-fuel 1))
            (set ship-mode 'avoiding)))))
    
    ; Mode-specific behavior
    (cond
      ; Patrol mode - circular pattern
      [(= ship-mode 'patrol)
       (let ([patrol-center [0 0]]
             [patrol-radius 5000])
         ; Calculate tangential velocity for circular motion
         (let ([to-center (vector-sub patrol-center ship-pos)]
               [dist-to-center (magnitude to-center)])
           (when (> (abs (- dist-to-center patrol-radius)) 100)
             ; Adjust radius
             (let ([radial-thrust (* 10 (- patrol-radius dist-to-center))])
               (apply-thrust ship (vector-scale to-center radial-thrust))
               (set ship-fuel (- ship-fuel 1))))))]
      
      ; Pursue mode - chase target
      [(= ship-mode 'pursue)
       (when ship-target
         (let ([target-pos (measure 'position ship-target)]
               [intercept-point (predict-intercept ship-pos ship-vel 
                                                  target-pos target-vel)])
           (let ([pursuit-vec (vector-sub intercept-point ship-pos)]
                 [thrust-mag (min 100 ship-fuel)])
             (apply-thrust ship (vector-scale pursuit-vec thrust-mag))
             (set ship-fuel (- ship-fuel thrust-mag))
             
             ; Check if caught
             (when (< (distance ship-pos target-pos) 100)
               (print "Target acquired!")
               (set ship-target nil)
               (set ship-mode 'patrol)))))]
      
      ; Flee mode - escape from threat
      [(= ship-mode 'flee)
       (let ([threats (query :property 'mass :min 10000)])
         (when threats
           (let ([escape-vec [0 0]])
             (loop threat threats
               (let ([threat-pos (measure 'position threat)]
                     [dist (distance ship-pos threat-pos)])
                 (when (< dist 2000)
                   (let ([away (vector-sub ship-pos threat-pos)])
                     (set escape-vec (vector-add escape-vec away))))))
             
             (when (> (magnitude escape-vec) 0)
               (apply-thrust ship (vector-scale escape-vec 200))
               (set ship-fuel (- ship-fuel 10))))))]
      
      ; Dock mode - approach station carefully
      [(= ship-mode 'dock)
       (when ship-target
         (let ([dock-pos (measure 'position ship-target)]
               [approach-dist (distance ship-pos dock-pos)]
               [approach-speed (magnitude ship-vel)])
           ; Slow approach
           (if (< approach-dist 200)
             ; Final docking
             (begin
               (let ([dock-vec (vector-sub dock-pos ship-pos)]
                     [max-speed 5])
                 (when (> approach-speed max-speed)
                   ; Brake
                   (apply-thrust ship (vector-scale ship-vel -10))
                   (set ship-fuel (- ship-fuel 2))))
               
               (when (and (< approach-dist 50) (< approach-speed 2))
                 (print "Docked successfully!")
                 (set ship-mode 'patrol)))
             ; Approach
             (let ([approach-vec (vector-sub dock-pos ship-pos)])
               (apply-thrust ship (vector-scale approach-vec 50))
               (set ship-fuel (- ship-fuel 5))))))]))
  
  ; Fuel management
  (when (< ship-fuel 100)
    (print "Low fuel! Seeking refuel station...")
    (set ship-mode 'dock)
    (set ship-target (find-nearest-station ship-pos)))
  
  ; Damage assessment
  (let ([broken-springs (count-broken-springs ship-hull)])
    (when (> broken-springs 3)
      (set ship-health (- ship-health (* broken-springs 5)))
      (print (format "Ship damaged! Health: ~a%" ship-health))
      (when (< ship-health 30)
        (set ship-mode 'flee)))))

; Helper functions for AI
(define (apply-thrust ship force-vec)
  (loop p (get-particles ship)
    (apply-force p (vector-scale force-vec (/ 1 (length (get-particles ship)))))))

(define (vector-sub v1 v2)
  [(- (first v1) (first v2))
   (- (second v1) (second v2))])

(define (vector-add v1 v2)
  [(+ (first v1) (first v2))
   (+ (second v1) (second v2))])

(define (vector-scale v s)
  [(* (first v) s)
   (* (second v) s)])

(define (magnitude v)
  (sqrt (+ (* (first v) (first v))
           (* (second v) (second v)))))

(define (predict-intercept ship-pos ship-vel target-pos target-vel)
  ; Simple linear prediction
  (let ([relative-pos (vector-sub target-pos ship-pos)]
        [relative-vel (vector-sub target-vel ship-vel)]
        [time-to-intercept (/ (magnitude relative-pos) 
                             (max 1 (magnitude relative-vel)))])
    (vector-add target-pos (vector-scale target-vel time-to-intercept))))

; Monitoring
(watch
  :every 0.5
  (print (format "Ship: mode=~a fuel=~a health=~a pos=~a"
                ship-mode ship-fuel ship-health 
                (measure 'center-of-mass ship-hull))))