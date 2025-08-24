; Star Lifecycle Simulation
; Models stellar evolution: formation, main sequence, red giant, and death

; Gas cloud that will form a star
(define gas-cloud
  (cloud
    :n 10000
    :center [0 0]
    :radius 1e12  ; Large diffuse cloud
    :mass-min 1e20
    :mass-max 1e22
    :temp 10))  ; Cold molecular cloud

; State variables for stellar evolution
(define star-phase 'molecular-cloud)
(define star-age 0)
(define star-core nil)
(define fusion-rate 0)
(define core-temp 10)
(define core-density 0)

; Main stellar evolution rule
(rule 'stellar-evolution
  :persistent true
  :update-interval 0.1
  
  (set star-age (+ star-age 0.1))
  
  (cond
    ; === MOLECULAR CLOUD PHASE ===
    [(= star-phase 'molecular-cloud)
     (let ([com (measure 'center-of-mass gas-cloud)]
           [cloud-radius (measure 'radius gas-cloud)]
           [density (/ (measure 'total-mass gas-cloud)
                      (* 4/3 pi (expt cloud-radius 3)))])
       
       (set core-density density)
       
       ; Check for gravitational instability (Jeans criterion)
       (when (> density 1e-10)  ; Threshold density
         (print "Gravitational collapse beginning...")
         (set star-phase 'collapsing)
         
         ; Add inward velocities to simulate collapse
         (loop p gas-cloud
           (let ([p-pos (measure 'position p)]
                 [collapse-vel (vector-scale (vector-sub com p-pos) -0.01)])
             (set-velocity p collapse-vel)))))]
    
    ; === COLLAPSING PHASE ===
    [(= star-phase 'collapsing)
     (let ([com (measure 'center-of-mass gas-cloud)]
           [avg-radius (measure 'average-distance-from com gas-cloud)]
           [total-ke (measure 'kinetic-energy gas-cloud)])
       
       ; Virial heating during collapse
       (set core-temp (* core-temp 1.01))
       
       ; Check for protostar formation
       (when (< avg-radius 1e10)  ; Sufficiently compact
         (print (format "Protostar formed! Core temp: ~a K" core-temp))
         (set star-phase 'protostar)
         
         ; Create dense core
         (set star-core
           (particle
             :mass (measure 'total-mass gas-cloud)
             :pos com
             :vel [0 0]
             :temp core-temp
             :radius 1e9))))]
    
    ; === PROTOSTAR PHASE ===
    [(= star-phase 'protostar)
     ; Continue accreting material
     (let ([nearby (find :center (measure 'position star-core) :radius 1e10)])
       (loop p nearby
         (when (and (!= p star-core) (< (random 1.0) 0.1))
           ; Accrete particle
           (set star-core.mass (+ star-core.mass p.mass))
           (set star-core.temp (* star-core.temp 1.001))
           (delete-particle p))))
     
     (set core-temp star-core.temp)
     
     ; Check for fusion ignition
     (when (> core-temp 1e7)  ; 10 million K
       (print "Fusion ignition! Main sequence star born!")
       (set star-phase 'main-sequence)
       (set fusion-rate 1.0)
       
       ; Clear remaining gas cloud
       (loop p gas-cloud
         (when (!= p star-core)
           (if (< (random 1.0) 0.5)
             ; Blow away in stellar wind
             (let ([wind-vel (vector-scale 
                             (vector-sub (measure 'position p)
                                       (measure 'position star-core))
                             100)])
               (set-velocity p wind-vel))
             ; Delete
             (delete-particle p))))))]
    
    ; === MAIN SEQUENCE PHASE ===
    [(= star-phase 'main-sequence)
     ; Stable hydrogen fusion
     (set fusion-rate (* 1.0 (expt (/ core-temp 1e7) 4)))  ; PP chain rate
     
     ; Energy generation creates outward pressure
     (let ([luminosity (* fusion-rate 3.828e26)])  ; Solar units
       
       ; Emit photons/stellar wind (simplified)
       (when (> (random 1.0) 0.9)
         (let ([angle (random 0 (* 2 pi))]
               [photon-energy 1e-10])
           (particle
             :mass photon-energy
             :pos (measure 'position star-core)
             :vel [(* 1000 (cos angle)) (* 1000 (sin angle))]
             :temp 0))))
     
     ; Slow fuel consumption
     (set star-core.mass (* star-core.mass 0.9999999))
     
     ; Solar flares
     (when (> (random 1.0) 0.98)
       (print "Solar flare!")
       (explode star-core :energy (* fusion-rate 1e20) :fragments 50))
     
     ; Check for fuel depletion (simplified)
     (when (< star-core.mass 1e29)  ; Lost significant mass
       (print "Hydrogen exhausted, entering red giant phase...")
       (set star-phase 'red-giant))]
    
    ; === RED GIANT PHASE ===
    [(= star-phase 'red-giant)
     ; Expand outer layers
     (set star-core.radius (* star-core.radius 1.001))
     (set star-core.temp (* star-core.temp 0.999))  ; Surface cooling
     
     ; Helium fusion in core (simplified)
     (set fusion-rate (* 0.1 (expt (/ core-temp 1e8) 3)))
     
     ; Mass loss through stellar wind
     (when (> (random 1.0) 0.8)
       (let ([wind-particles 10])
         (loop i 0 wind-particles
           (let ([angle (random 0 (* 2 pi))]
                 [speed (random 100 500)])
             (particle
               :mass (/ star-core.mass 1e6)
               :pos (measure 'position star-core)
               :vel [(* speed (cos angle)) (* speed (sin angle))]
               :temp 3000)))))
     
     (set star-core.mass (* star-core.mass 0.99999))
     
     ; Check for final phase
     (when (or (< star-core.mass 5e28)  ; Too much mass lost
               (> star-age 1000))      ; Or too old
       (if (> star-core.mass 1.4e30)  ; Chandrasekhar limit
         (begin
           (print "Core collapse! Supernova imminent!")
           (set star-phase 'supernova))
         (begin
           (print "Gentle death, becoming white dwarf...")
           (set star-phase 'white-dwarf))))]
    
    ; === SUPERNOVA PHASE ===
    [(= star-phase 'supernova)
     (print "SUPERNOVA EXPLOSION!")
     
     ; Massive explosion
     (explode star-core 
       :energy (* star-core.mass 1e44)  ; ~1% mass to energy
       :fragments 1000)
     
     ; Create neutron star or black hole remnant
     (if (> star-core.mass 3e30)
       (begin
         (print "Black hole formed!")
         (particle
           :mass (* star-core.mass 0.1)  ; Remnant mass
           :pos (measure 'position star-core)
           :vel [0 0]
           :temp 0
           :type 'black-hole))
       (begin
         (print "Neutron star formed!")
         (particle
           :mass (* star-core.mass 0.1)
           :pos (measure 'position star-core)
           :vel [0 0]
           :temp 1e9
           :radius 10000)))  ; 10km radius
     
     (delete-particle star-core)
     (set star-phase 'remnant)]
    
    ; === WHITE DWARF PHASE ===
    [(= star-phase 'white-dwarf)
     ; Slowly cooling degenerate matter
     (set star-core.temp (* star-core.temp 0.9999))
     (set star-core.radius 5000000)  ; Earth-sized
     
     ; Occasional nova if accreting
     (let ([nearby (find :center (measure 'position star-core) :radius 1e8)])
       (when (> (length nearby) 1)
         (when (> (random 1.0) 0.99)
           (print "Nova eruption!")
           (explode star-core :energy 1e35 :fragments 20))))]
    
    ; === REMNANT PHASE ===
    [(= star-phase 'remnant)
     ; Simulation complete
     (print (format "Star lifecycle complete. Total age: ~a" star-age))]))

; Monitor stellar properties
(watch
  :every 1.0
  (print (format "Phase: ~a Age: ~a Core-T: ~a K Fusion: ~a Mass: ~a"
                star-phase star-age core-temp fusion-rate
                (if star-core (measure 'mass star-core) 0))))